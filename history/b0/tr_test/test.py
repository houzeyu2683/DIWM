from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer, BertModel

'''
tokenizer.add_special_tokens(["[XXX]"])
可以對 tokenizer 添加特殊的 token 符號。
然後你最終還是會出現 error 
你要參考下面
special_tokens_dict = {'additional_special_tokens': ['[C1]','[C2]','[C3]','[C4]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

也許下面可以獲得對應的 word embedding 。
words = bert_model("This is an apple")
word_vectors = [w.vector for w in words]
'''




import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# special_tokens_dict = {'additional_special_tokens': ['[C1]','[C2]','[C3]','<start>']}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

model = BertModel.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))
input_ids = torch.tensor(tokenizer.encode("sadsadsa [C7] Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
input_ids.shape
outputs = model(input_ids)
len(outputs)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
last_hidden_states.shape
last_hidden_states.transpose(0, 1)


tokenizer.decode(['3736'])



input_ids = torch.tensor(tokenizer.encode("Hello, my <start>")).unsqueeze(0)
tokenizer.decode(['30525'])
