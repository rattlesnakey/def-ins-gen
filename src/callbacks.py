import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from utils import save_json


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #! np.prod 是把 list 里面的乘起来，比如 size = 1,2,3，那么乘起来就是 1x2x3 = 6个参数
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)

#! 我觉得可以不用它这个LoggingCallback, 用自带的就好了
# class Seq2SeqLoggingCallback(pl.Callback):
#     def on_batch_end(self, trainer, pl_module):
#         lrs = {f"lr_group_{i}": param["lr"] for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)}
#         pl_module.logger.log_metrics(lrs)

#     @rank_zero_only
#     def _write_logs(
#         self, trainer: pl.Trainer, pl_module: pl.LightningModule, type_path: str, save_generations=True
#     ) -> None:
#         logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
#         #! test_epoch_end 返回的字典就是这里的 metric
#         metrics = trainer.callback_metrics
#         trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]})
#         # Log results
#         od = Path(pl_module.hparams.output_dir)
#         if type_path == "test":
#             results_file = od / "test_results.txt"
#             # generations_file = od / "test_generations.txt"
#         else:
#             # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
#             # If people want this it will be easy enough to add back.
#             results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
#             generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
#             results_file.parent.mkdir(exist_ok=True)
#             generations_file.parent.mkdir(exist_ok=True)
            
#         #! 这里的 metrics 是 trainer.callback_metrics 返回的，要改成把 bleu 这些也都返回才行
#         with open(results_file, "a+") as writer:
#             for key in sorted(metrics):
#                 if key in ["log", "progress_bar", "preds"]:
#                     continue
#                 val = metrics[key]
#                 if isinstance(val, torch.Tensor):
#                     val = val.item()
#                 msg = f"{key}: {val:.6f}\n"
#                 writer.write(msg)

#         if not save_generations:
#             return

#         if "preds" in metrics:
#             content = "\n".join(metrics["preds"])
#             generations_file.open("w+").write(content)

#     @rank_zero_only
#     def on_train_start(self, trainer, pl_module):
#         try:
#             npars = pl_module.model.model.num_parameters()
#         except AttributeError:
#             npars = pl_module.model.num_parameters()

#         n_trainable_pars = count_trainable_parameters(pl_module)
#         # mp stands for million parameters
#         trainer.logger.log_metrics({"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6})
#     #! 改了一下这里，在 test 和 validation end 的时候加了 save_json
#     #! 就是把训练过程中所有中间的 Metric 值保存下来
#     @rank_zero_only
#     def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
#         # save_json(pl_module.metrics, pl_module.metrics_save_path)
#         return self._write_logs(trainer, pl_module, "test")

#     @rank_zero_only
#     def on_validation_end(self, trainer: pl.Trainer, pl_module):
#         pass
#         # save_json(pl_module.metrics, pl_module.metrics_save_path)
#         # Uncommenting this will save val generations
#         # return self._write_logs(trainer, pl_module, "valid")

#! 在 main 函数里面有调用这个函数，前面两个参数是 main 里面传进去的
def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{epoch_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{epoch_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{epoch_count}"
    elif metric == 'bleu_and_nist':
        exp = "{val_avg_bleu_and_nist:.4f}-{epoch_count}"
        
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )
    #! 这里是在定义路径的时候，用到了其他变量
    checkpoint_callback = ModelCheckpoint(
        #! 这个是 checkpoint 保存的路径及名称
        #! monitor 的值应该是 validation_epoch_end 的值
        filepath=os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )
