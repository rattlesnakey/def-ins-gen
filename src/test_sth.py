from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
def model_test():
    model = T5ForConditionalGeneration.from_pretrained('../pretrained_models/t5')
    tokenizer = T5Tokenizer.from_pretrained('../pretrained_models/t5')
    # model.save_pretrained('../pretrained_models/t5')
    # tokenizer.save_pretrained('../pretrained_models/t5')


    text = '<definition> I love apple <instance> <extra_id_0>'
    text2 = '<instance> apple in context "<extra_id_0>" means fruit'
    encoded_text2 = tokenizer.tokenize(text2)
    num_added_tokens = tokenizer.add_tokens(['<definition>', '<instance>'])

    model.resize_token_embeddings(len(tokenizer))


    encoded_text2 = tokenizer.batch_encode_plus([text2])
    print(encoded_text2)
    input_ids = torch.tensor(encoded_text2['input_ids'])

    output_ids = model.generate(input_ids)

    print(output_ids)
    output = tokenizer.decode(output_ids[0], skip_special_token=True)
    print(output)
    # tokenizer.add_special_tokens()
    
#! 只是把句号去掉了而已
def get_rid_of_period(s):
    pattern = re.compile("\.(?!\d)")
    return pattern.sub('', s)

def metric_test():
    from nltk.translate import bleu_score, nist_score
    x = 'I am happy !.'
    y = get_rid_of_period(x)
    print(y)
    pred = ['apple', 'happy', '?']
    #! refs 越多，nist 就会越高
    #! bleu 如果每个 ref 都有匹配一部分的话，也会增高
    refs = [['I' ,'eat', 'apple'], ['I', "eat", 'happy', 'test']]
    n = 5
    if len(pred) < 5:
        n = len(pred)
    #! auto_reweigh 好像一直都是 True, 之前搞错了
    #! 之前有 zero 的错误，是因为 n 没搞清楚
    auto_reweigh = False if len(pred) == 0 else True
    print(auto_reweigh)
    nist = nist_score.sentence_nist(refs, pred, n=n)
    bleu = bleu_score.sentence_bleu(refs, pred, smoothing_function=bleu_score.SmoothingFunction().method2, auto_reweigh=auto_reweigh)
    print({'nist':nist, 'bleu':bleu})
    
    
metric_test()