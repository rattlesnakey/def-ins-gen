import argparse
import glob
from linecache import cache
import logging
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import json
from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, AdamW
# from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import CrossEntropyLoss
from nltk.translate import bleu_score, nist_score

try:
    from transformers import MT5ForConditionalGeneration
    from .utils import (
        assert_all_frozen,
        lmap,
        flatten_list,
        pickle_save,
#         save_git_info,
        save_json,
        freeze_params,
        # calculate_rouge,
#         get_git_info,
        ROUGE_KEYS,
        Seq2SeqDataset,
    )

    from .callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
except ImportError:
    from utils import (
        Seq2SeqDataset,
        assert_all_frozen,
        lmap,
        flatten_list,
        pickle_save,
#         save_git_info,
        save_json,
        freeze_params,
#       calculate_rouge,
#         get_git_info,
        ROUGE_KEYS,
    )
    from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42) 
logger = logging.getLogger(__name__)

class SummarizationModule(pl.LightningModule):
    #! loss_names 和 metric_names 在这里
    #! 这边的这些属性其实也可以直接 self. 来引用
    loss_names = ["loss"]
    metric_names = ["bleu", 'nist']
    val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super(SummarizationModule, self).__init__()
        #! 把 
        self.hparams = hparams
        self.output_dir = self.hparams.output_dir
        if 'mt5' in hparams.model_name_or_path:
            pass
#             self.model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        else:
            #! 这边加了 cache_dir
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path, cache_dir='../pretrained_models/t5')
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.model_name_or_path, cache_dir='../pretrained_models/t5')
        
#         save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.hparams.output_dir) / "metrics.json"
        # self.hparams_save_path = Path(self.hparams.output_dir) / "hparams.json"
        # print(self.hparams)
        #! 保存超参数
        # save_json(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        
        #! data_dir 这里传进来了
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
        )
        self.option = self.hparams.option
        self.resume_ckpt = self.hparams.resume_ckpt
    
        self.sample = self.hparams.sample
        self.beams_penalty = self.hparams.beams_penalty
        self.beams_group = self.hparams.beams_group
        self.num_beams = self.hparams.num_beams
        self.type_path = ''
        #! test_val 应该是拿val集来测试
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
            "test_val": self.hparams.n_test,            
        }
        
        #! 相当于把 n_observation_per_split 复制了一个过来..
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
        
        #! 这边有传进来每个的最大长度
        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
            "test_val": self.hparams.test_max_target_length,            
        }
        
        #! 不知道为什么一定要小于等于
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        
        #! 这两个是自己定义的而已，不是 lightning 的组件
        #! 这些好像代码有问题，反正也不执行
        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

#         self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None
        #! 把 Dataclass 传进来了
        self.dataset_class = Seq2SeqDataset

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    #! 在 generative 的时候会使用，把 special token 去掉了
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        #! 这里用 strip map 了每个list
        return lmap(str.strip, gen_text)

    
    def _step(self, batch: dict) -> Tuple:

        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        
        decoder_input_ids = self.model._shift_right(target_ids)
        lm_labels = target_ids
        #! 这个默认就是调用 self.forward 函数哈
        outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False)

        # Same behavior as modeling_bart.py
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        lm_logits = outputs[0]
        assert lm_logits.shape[-1] == self.model.config.vocab_size
        loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), lm_labels.view(-1))
        
        return (loss,)
    #! 这个没用
    # @property
    # def pad(self) -> int:
    #     return self.tokenizer.pad_token_id
    
    #! 返回 loss dict
    def training_step(self, batch, batch_idx) -> Dict:
        #! loss_tensors:(loss,)
        #! 这边其实只有 loss 而已
        loss_tensors = self._step(batch)
        #! logs:{'loss':value}
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        #! 其实就只有 loss
        return {"loss": loss_tensors[0], "log": logs}
    
    #! 生成文本
    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)
    
    #! 返回指标
    #! outputs 是所有 batch 对应的 loss、bleu、nist
    def validation_epoch_end(self, outputs) -> Dict:
        self.step_count += 1
        #! 这边只有 loss
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        #! 把 loss 取出来
        loss = losses["loss"]
#         rouges = {k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]}
        #! metric_names 有 bleu、nist
        metrics = {k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names}
        
        val_metric_tensor: torch.FloatTensor = torch.tensor(metrics[self.val_metric]).type_as(loss)
        metrics.update({k: v.item() for k, v in losses.items()})
        # losses.update(rouges)
        # metrics = {f"val_avg_{k}": x for k, x in losses.items()}
        #! 记录是第几步骤
        metrics["step_count"] = self.step_count
        # print(metrics)
        #! metric 写到文件里
        self.save_metrics(metrics, 'val')  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {"log": metrics, "preds": preds, "val_loss": loss, f"val_{self.val_metric}": val_metric_tensor}
    
    #! 在 validation_epoch_end 被调用
    def save_metrics(self, latest_metrics, type_path) -> None:
        #! {val:[metrics]}
        #! 相当于是一个字典的value 是一个字典组成的 list
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)
        
    #! 这个用在 generative_step 那边
    def calculate_bleu_and_nist(self, preds, targets):  
        # if fast:
        #     return {'bleu':self.step_count}
        avg_bleu, avg_nist = [], []
        #! 这边就是把 bleu 值算出来
        for item in zip(preds, targets):
            pred = item[0].split() if len(item[0].split())>0 else ['dummy']
            tar = item[1].split() if len(item[1].split())>0 else ['dummy']
            bleu = bleu_score.sentence_bleu([tar], pred,
                            smoothing_function=bleu_score.SmoothingFunction().method2, auto_reweigh=True)
            #! nist 分数 nltk 支持好像有点问题
            try:
                nist = nist_score.sentence_nist([tar], pred,)
            except ZeroDivisionError: 
                nist = 0
            avg_bleu.append(bleu)
            avg_nist.append(nist)
            
        #! 这边不应该加1，+1被我去掉了
        return {'bleu':sum(avg_bleu)/(len(avg_bleu)), 'nist':sum(avg_nist)/(len(avg_nist))}
    #! 这个函数没用
    # def calc_generative_metrics(self, preds, target) -> Dict:
    #     return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        #! 这个 fast 应该是 teacher_forcing 
        # fast = False
        # generated_ids = batch["decoder_input_ids"]
        # if not fast:
        
        generated_ids = self.model.generate(
            batch["input_ids"],
            #! 这个还不知道是啥意思
            attention_mask=batch["attention_mask"],
            use_cache=True,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        #! 就是把解码结果中的 special token 去除掉
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._step(batch)
        #! 这边只有 loss
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
#         rouge: Dict = self.calc_generative_metrics(preds, target)
        #! 在generative 里面用了 calculate_bleu_and_nist
        rouge: Dict = self.calculate_bleu_and_nist(preds, target)
        #! summary_len 也就是解码的长度
        summ_len = np.mean(lmap(len, generated_ids))
        # base_metrics.update(preds=preds, target=target, **rouge)
        #! 返回一个 dict 指标
        #! 这边有 rouge, rouge 里面有 bleu 和 nist
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge)            
        return base_metrics
    
    #! 这个就是相当于之前自己写的 predict.py 一样，在这边调用 model.generate 和 tokenizer.decode
    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]
        #! 如果是释义生成任务的话，那么生成的释义里面不要包含被释义词
        bad_word_list = batch["target_word"]

#         loss = self._step(batch)
        # loss = torch.tensor(0.0)
        # losses = []
        # log_list = []
        # avg_list = []
        # sum_list = []        
        #! 如果要测试的话，就用 forward
        #if (not args.option.startswith('t5_specific') and not args.option.startswith('forward') and not args.option.startswith('t5_general')):
        #! 这边加了一鞋策略
        #! 到时候可以改一下
        if self.hparams.task == 'def-gen':
            generated_ids = self.model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    num_beams=self.num_beams,
                    repetition_penalty=1.0,
                    length_penalty=0.8,        
                    no_repeat_ngram_size=1,
                    early_stopping=True,
                    use_cache=True,
                    bad_words_ids=bad_word_list,
                    #! return_sequences 改成 1
                    num_return_sequences=1  
                )
        #! 区别在于 bad_word 用不用
        elif self.hparams.task  == 'ins-gen':
            generated_ids = self.model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    num_beams=self.num_beams,
                    repetition_penalty=1.0,
                    length_penalty=0.8,        
                    no_repeat_ngram_size=1,
                    early_stopping=True,
                    use_cache=True,
                    # bad_words_ids = bad_word_list,
                    num_return_sequences=1  
                )
        #! 这边是 num_beams 个结果，所以得设置成1
        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        #! t只是batch 里面的一个句子
        targets = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        
        inputs = [
            self.tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for s in source_ids
        ]
        #! preds 和 targets 都是一个 batch 
        #! bleu 和 nist
        metrics = self.calculate_bleu_and_nist(preds, targets)
        base_output = {'inputs':inputs, 'preds':preds, 'targets':targets}
        # #! 这是算 train 的时候的 loss 的
        # else:
        #     #! option 等于空的时候是这样的
        #     source_ids, source_mask, target_ids = batch["input_ids"], batch["attention_mask"], batch["decoder_input_ids"]            
        #     decoder_input_ids = self.model._shift_right(target_ids)
        #     #! 这个是用的 teacher_forcing 了
        #     outputs = self(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        #     loss_fct = CrossEntropyLoss(ignore_index=pad_token_id)
        #     labels_hat = torch.argmax(outputs[0], dim=2)
        #     preds = [
        #         self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         for g in labels_hat
        #     ]
        #     targets = [
        #         self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #         for g in target_ids
        #     ]       
        #     m = torch.nn.LogSoftmax(dim=2) 
        #     after_logsoftmax = m(outputs[0])         
        #     for i in range(len(outputs[0])):
        #         # calculate cross entropy by sentence not by batch
        #         loss = loss_fct(outputs[0][i].view(-1, outputs[0][i].size(-1)), target_ids[i].cuda().view(-1))

        #         for j in range(len(after_logsoftmax[i])):
        #             log_list.append(after_logsoftmax[i][j][target_ids[i][j].item()].item())
        #             if target_ids[i][j].item()==1:
        #                 break
        #         avg_list.append(str(sum(log_list)/(len(log_list)+0.001)))
        #         sum_list.append(str(sum(log_list)))
        #         log_list = []                    
        #         losses.append(str(loss.item()))
        # {"test_loss": loss, "preds": preds, "targets": targets, "losses": losses, "log_sum": sum_list, "log_avg": avg_list}
        # print(base_output)
        # print(metrics)
        #! 土办法合并
        for k, v in metrics.items():
            base_output[k] = v
        # output = dict(base_output.items() + metrics.items())
        # print(base_output)
        #! update 有问题
        # output = base_output.update(metrics)
        return base_output
    
    
#     #! 多卡的时候才会调用，所以一张卡的时候没有调用
#     def test_end(self, outputs):
#         output_test_predictions_file = os.path.join(
#             #! test_dataset 传 test_val 的时候，返回的就是 val_predictions，因为取的是 val
#             args.output_dir, "{}_predictions.txt".format(self.hparams.test_dataset.split('_')[-1]))
#         # output_test_targets_file = os.path.join(
#         #     args.output_dir, "{}_targets.txt".format(self.hparams.test_dataset.split('_')[-1]))
#         # output_test_losses_file = os.path.join(
#         #     args.output_dir, "{}_losses.txt".format(self.hparams.test_dataset.split('_')[-1]))

#         # write predictions and targets
#         # if self.option=="":
#             # output predictions
#         #! 把结果保存
#         with open(output_test_predictions_file, "w",encoding='utf-8') as p_writer:
#             for output_batch in outputs:
#                 for inp, pred, tgt in zip(output_batch["inputs"], output_batch["preds"], output_batch["targets"]):
#                     p_writer.writelines(inp + '\t' + pred + '\t' + tgt + "\n")
#             p_writer.close()

#         # else:
#         # # output loss(scores)            
#         #     with open(output_test_losses_file, "w",encoding='utf-8') as r_writer:
#         #         for output_batch in outputs:
#         #             r_writer.writelines(s + "\n" for s in output_batch["losses"])
#         #         r_writer.close()

# #         # output targets
# #         with open(output_test_targets_file, "w",encoding='utf-8') as t_writer:
# #             for output_batch in outputs:            
# #                 t_writer.writelines(s + "\n" for s in output_batch["targets"])
# #             t_writer.close()
#        return self.test_epoch_end(outputs)
    
    #! 只算了 test_loss 
    def test_epoch_end(self, outputs):
        output_test_predictions_file = os.path.join(
            #! test_dataset 传 test_val 的时候，返回的就是 val_predictions，因为取的是 val
            args.output_dir, "{}_predictions.txt".format(self.hparams.test_dataset.split('_')[-1]))
        #! 把结果保存
        with open(output_test_predictions_file, "w",encoding='utf-8') as p_writer:
            for output_batch in outputs:
                for inp, pred, tgt in zip(output_batch["inputs"], output_batch["preds"], output_batch["targets"]):
                    p_writer.writelines(json.dumps({'inp':inp, 'pred':pred, 'tgt':tgt}) + '\n')
            p_writer.close()
            
        # np.array([x[k] for x in outputs]).mean()
        avg_bleu = np.array([x["bleu"] for x in outputs]).mean()
        avg_nist = np.array([x["nist"] for x in outputs]).mean()
        metric = {"avg_test_bleu": avg_bleu, "avg_test_nist":avg_nist}
        save_json(metric, Path(self.hparams.output_dir) / "test_result.json")
        return {"avg_test_bleu": avg_bleu, "avg_test_nist":avg_nist}
    
    
    
    #! 这里构建 dataset 类
    #! type_path 有 train, val, test
    #! option 有 general, specific
    def get_dataset(self, type_path, option, self_ref) -> Seq2SeqDataset:
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            max_target_length=max_target_length,
            option = option,
            self_ref = self_ref,
            task=self.hparams.task,
            sample=self.hparams.sample,  #! 正式训练的时候不要传 sample 这个参数
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path, self.hparams.option, self.hparams.self_ref)
        #! 这边有 collate_fn 我可以自己改
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,  #! cpu 跑的时候，这个要改成1，要不会报错
            sampler=None,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        if max(scheduler.get_last_lr()) > 0:
            warnings.warn("All learning rates are 0")
        #! 这个 sheduler 没用到
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        #! 这个地方得改成 valid
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        #! 从外面的 args 传 test_dataset 路径进来
        #! test_dataset 是外面指定的
        self.type_path = self.hparams.test_dataset
        return self.get_dataloader(self.type_path, batch_size=self.hparams.eval_batch_size)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]
    
    @staticmethod
    def add_model_specific_args(parser, root_dir):
        #! 这边 add 好像没有返回回来
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)

        parser.add_argument(
            "--max_source_length",
            default=200,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=150,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=150,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=150,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )     
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--logger_name", type=str, choices=["default"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser
    
    
def main(args, model=None) -> SummarizationModule:
    
    Path(args.output_dir).mkdir(exist_ok=True)
    model: SummarizationModule = SummarizationModule(args) 
            
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
        
    ck = False
    if args.resume_ckpt:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
        ck = checkpoints[-1]
    #! 这里是用 bleu metric 来处理的
    es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    
    #! 没有传 dataset，dataset 和 dataloader 都是直接封装在 model 类里面
    #! 这边 generic_train 里面就有执行 trainer.fit 了
    #! extra_callbacks 加一下 pretty metric table
    # from pl_bolts.callbacks import PrintTableMetricsCallback
    # callback = PrintTableMetricsCallback()
    # args.tpu_cores = 0
    #! tpu_cores 是一个 function，json 没办法保存
    model.hparams.tpu_cores = 0
    save_json(model.hparams, args.output_dir+"hparams.json")
    trainer: pl.Trainer = generic_train(
        model,
        args,
        # logging_callback=Seq2SeqLoggingCallback(),  #! 把这个 logging callback 换成系统自带的试试
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        early_stopping_callback=es_callback,
        resume_from_checkpoint=ck,
        logger=logger,
    )   
    #! 汇报模型的超参数保存下来
    
    
    #! predict 的话也是读取最好的 ckpt 进来进行 test
    #! 可以同时 --do_train 和 --do_predict
    # if not args.do_predict:
    #     return model
    # print(args.do_predict)
    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    print(checkpoints[-1])
    # trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test(model)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    #! os.getcwd() 是当前的 root dir, 所以会把模型文件夹放在当前的目录下
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    #! 加 task
    
    parser.add_argument(
            "--task",
            type=str,
            default = 'def-gen',
            help="def-gen or ins-gen",
        )
    parser.add_argument(
            "--option",
            type=str,
            default = '',
            help="t5_general or t5_specific",
        )
    parser.add_argument(
            "--self_ref",
            action="store_false",
            default = True,
            help="containing self-reference or not",
        )      
    parser.add_argument(
            "--resume_ckpt",
            action="store_true",
            default = False,
            help="resume training",
        )  
    #! 不传 test_dataset 的时候，默认是 test
    parser.add_argument(
            "--test_dataset",
            type=str,
            default = 'test',
            help="generate prediction for test set or validation set ",
        )   
    parser.add_argument(
            "--sample",
            type=int,
            default=None,
            help="how many samples are used",
        )       
    parser.add_argument(
            "--beams_penalty",
            type=float,
            default = 1.0,
            help="penalty for diverse beam search",
        ) 
    parser.add_argument(
            "--beams_group",
            type=int,
            default = 1,
            help="how many group of beams",
        )    
    parser.add_argument(
            "--num_beams",
            type=int,
            default = 100,
            help="The number of beam search",
        )      
    args = parser.parse_args()

    main(args)
