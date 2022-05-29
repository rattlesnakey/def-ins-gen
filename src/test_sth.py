from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='../pretrained_models/t5')
tokenizer = T5Tokenizer.from_pretrained('t5-small', cache_dir='../pretrained_models/t5')

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