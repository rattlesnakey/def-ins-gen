
import sys
import os
import json
prediction_path = os.path.join(sys.argv[1], 'test_def_gen_predictions.txt')
ins_gen_predition_path = os.path.join(sys.argv[1], 'test_ins_gen_predictions.txt')
save_metric_path = os.path.join(sys.argv[1], 'final_test_metric.json')


f = open(prediction_path, 'r')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist
import random
import string
import os

from subprocess import Popen, PIPE


tmp_dir = "./tmp"
os.makedirs(tmp_dir, exist_ok=True)
suffix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
hyp_path = os.path.join(tmp_dir, 'hyp-' + suffix)
base_ref_path = os.path.join(tmp_dir, 'ref-' + suffix)
to_be_deleted = set()
to_be_deleted.add(hyp_path)
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir='../pretrained_models/t5')


def bleu(pred, targets, smoothing_function=SmoothingFunction().method2):
    return sentence_bleu(targets, pred, smoothing_function=smoothing_function, auto_reweigh=True)

def nist(pred, targets):
    n = 5
    if len(pred) < 5:
        n = len(pred)
    return sentence_nist(targets, pred, n)

def bleu_cpp(pred, targets):
    ref_paths = []
    for i, ref in enumerate(targets):
        ref_path = base_ref_path + str(i)
        with open(ref_path, 'w+') as f:
            f.write(' '.join(ref) + '\n')
            ref_paths.append(ref_path)
            to_be_deleted.add(ref_path)
    with open(hyp_path, 'w+') as f:
        f.write(' '.join(pred) + '\n')
    
    bleu_path = './sentence-bleu' 
    rp = Popen(['cat', hyp_path], stdout=PIPE)
    bp = Popen([bleu_path] + ref_paths, stdin=rp.stdout, stdout=PIPE)
    out, err = bp.communicate()
    bleu = float(out.strip())
    return bleu


ref_dict = {}

import re
def get_word(inp):
    word = re.findall("(?<=word: )([\w]*)(?= context:)", inp)[0] 
    return word

def build_ref_dict(inps, preds, tgts, words):
    for inp, pred, tgt, word in zip(inps, preds, tgts, words):
        # word = re.findall("(?<=word: )([\w]*)(?= context:)", inp)[0] 
        word_list = tokenizer.convert_ids_to_tokens(word)
        # print(word_list)
        word = ' '.join(word_list)
        # print(word)
        if word not in ref_dict:
            ref_dict[word] = []
        ref_dict[word].append(tgt)


import json
bleus_single, bleus_multi = [], []
nists_single, nists_multi = [], []
bleus_cpp_multi = []
inps, preds, tgts, words = [], [], [], []
for line in f:
    line = line.strip()
    line = json.loads(line)
    inp, pred, tgt, word = line['inp'], line['pred'], line['tgt'], line['word']
    inps.append(inp); preds.append(pred); tgts.append(tgt); words.append(word)

build_ref_dict(inps, preds, tgts, words)

for inp, pred, tgt, word in zip(inps, preds, tgts, words):
    
    # word = get_word(inp)
    word_list = tokenizer.convert_ids_to_tokens(word)
    word = ' '.join(word_list)
    tgts_multi = ref_dict[word]
    tgts_single = [tgt]
    pred = pred.strip().split()
    tgts_multi = [tgt.strip().split() for tgt in tgts_multi]
    tgts_single = [tgt.strip().split() for tgt in tgts_single]
    
    cur_bleu_single = bleu(pred, tgts_single)
    cur_nist_single = nist(pred, tgts_single)
    
    cur_bleu_cpp_multi = bleu_cpp(pred, tgts_multi)
    
    
    cur_bleu_multi = bleu(pred, tgts_multi)
    cur_nist_multi = nist(pred, tgts_multi)
    
    bleus_single.append(cur_bleu_single)
    nists_single.append(cur_nist_single)
    bleus_multi.append(cur_bleu_multi)
    nists_multi.append(cur_nist_multi)
    bleus_cpp_multi.append(cur_bleu_cpp_multi) 

    

one2one_bleu_nltk = sum(bleus_single)/len(bleus_single)
one2one_nist_nltk = sum(nists_single)/len(nists_single)


one2many_bleu_nltk = sum(bleus_multi)/len(bleus_multi)
one2many_nist_nltk = sum(nists_multi)/len(nists_multi)

one2many_bleu_cpp = sum(bleus_cpp_multi)/len(bleus_cpp_multi)

result = {'one2one_bleu_nltk':one2one_bleu_nltk, 'one2one_nist_nltk':one2one_nist_nltk, 'one2many_bleu_nltk':one2many_bleu_nltk, 'one2many_nist_nltk':one2many_nist_nltk, 'one2many_bleu_cpp':one2many_bleu_cpp}

json.dump(result, open(save_metric_path, 'w+'))
print('done')
