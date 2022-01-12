

import math
from typing import ForwardRef
import numpy
import torch
import torchvision
from torch import nn


constant = {
    'image embedding size':256,
    'text embedding size':256
}


# class position(nn.Module):

#     def __init__(self, emb_size=256, dropout=0.1, maxlen=5000):

#         super(position, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#         self.dropout = nn.Dropout(dropout)
#         # self.register_buffer('pos_embedding', pos_embedding)
#         self.pos_embedding = pos_embedding

#     def forward(self, x):

#         y = self.dropout(x + self.pos_embedding[:x.size(0), :])
#         return(y)

class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        layer = {}
        layer['01'] = cross(self.vocabulary)
        self.layer = nn.ModuleDict(layer)
        pass
    
    def forward(self, image=None, text=None, device='cpu'):

        self.layer = self.layer.to(device)
        value = {}
        image, text = image.to(device), text.to(device)
        value['01'] = self.layer['01']([image, text])
        y = value['01']
        return(y)

    def autoregressive(self, image=None, device='cpu', limit=20):

        self.layer = self.layer.to(device)
        image = image.to(device)
        generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)        
        # generation = torch.full((1, len(image)), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
        # value['output'] = []
        # value['image embedding'] = self.layer['image embedding'](image)
        # print("value['image embedding']")
        # print(value['image embedding'].shape)
        value = {}
        for _ in range(limit-1):
            
            with torch.no_grad():

                value['next word probability'] = self.layer['01']([image, generation])
                value['next word probability'] = value['next word probability'][-1,:,:]
                value['next word prediction']  = value['next word probability'].argmax(axis=1).view((1,1))
                generation = torch.cat([generation, value['next word prediction']], dim=0)
                if(value['next word prediction'] == self.vocabulary.index['<end>']): break
                pass

            pass

        y = list(generation.view(-1).cpu().numpy())
        return(y)

    pass

class cross(nn.Module):

    def __init__(self, vocabulary=None):

        super(cross, self).__init__()
        self.vocabulary = vocabulary
        layer = {}
        layer['01'] = image()
        layer['02'] = text(vocabulary=self.vocabulary)
        layer['03'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=constant['text embedding size'], nhead=4),
            num_layers=2,
            norm=None
        )
        layer['04'] = nn.Linear(constant['text embedding size'], self.vocabulary.size)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        value = {}
        value['image'], value['text'] = x
        value['01'] = self.layer['01'](value['image'])
        value['02'] = self.layer['02'](value['text'])
        value['03'] = self.layer['03'](
            tgt=value['02'], memory=value['01'], 
            tgt_mask = mask().sequence(value['text'], "autoregressive"),
            memory_mask = None, 
            tgt_key_padding_mask = mask().padding(value['text'], self.vocabulary.index['<padding>']),
            memory_key_padding_mask = None
        )
        value['04'] = self.layer['04'](value['03'])
        y = value['04']
        return(y)

    pass


# class model(nn.Module):

#     def __init__(self, vocabulary=None):

#         super(model, self).__init__()
#         self.vocabulary = vocabulary
#         self.mask = {"decoder":mask(mode='decoder')}
#         layer = {}
#         layer['image embedding'] = image()
#         layer['text embedding']  = text(vocabulary=self.vocabulary)
#         # layer['text position']   = position()
#         layer['text encoder'] = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(
#                 d_model=constant['text embedding size'], nhead=4, 
#                 dim_feedforward=2048, dropout=0.1, activation='relu'
#             ),
#             num_layers=2, norm=None
#         )
#         layer['image and text decoder'] = nn.TransformerDecoder(
#             decoder_layer=nn.TransformerDecoderLayer(
#                 d_model=constant['text embedding size'], 
#                 nhead=4, dim_feedforward=2048, dropout=0.1, activation='relu'
#             ), 
#             num_layers=3, norm=None
#         )
#         layer['output'] = nn.Linear(constant['text embedding size'], self.vocabulary.size)
#         self.layer = nn.ModuleDict(layer)        
#         return

#     def forward(self, image=None, text=None, device='cpu'):

#         image = image.to(device)
#         text  = text.to(device)
#         pass

#         self.layer = self.layer.to(device)
#         value = {}
#         value['image embedding']     = self.layer['image embedding'](image)
#         # value['text embedding']      = self.layer['text position'](self.layer['text embedding'](text))
#         value['text embedding']      = self.layer['text embedding'](text)
#         value['text attention mask'] = self.mask['decoder'].sequence(x=text)
#         value['text padding mask']   = self.mask['decoder'].padding(x=text, value=self.vocabulary.index['<padding>'])
#         value['text embedding']      = self.layer['text encoder'](value['text embedding'], mask=value['text attention mask'], src_key_padding_mask=value['text padding mask'])
#         value['image and text decoder'] = self.layer['image and text decoder'](
#             tgt=value['text embedding'], memory=value['image embedding'], 
#             tgt_mask=value['text attention mask'], memory_mask=None, 
#             tgt_key_padding_mask=value['text padding mask'], memory_key_padding_mask=None
#         )
#         value['output'] = self.layer['output'](value['image and text decoder'])
#         return(value['output'])

#     def autoregressive(self, image=None, device='cpu', limit=20):

#         image = image.to(device)
#         generation = torch.full((1, len(image)), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
#         pass

#         self.layer = self.layer.to(device)
#         value = {}
#         value['output'] = []
#         value['image embedding'] = self.layer['image embedding'](image)
#         # print("value['image embedding']")
#         # print(value['image embedding'].shape)
#         for _ in range(limit-1):

#             value['text embedding']         = self.layer['text embedding'](generation)
#             value['text attention mask']    = self.mask['decoder'].sequence(x=generation)
#             value['image and text decoder'] = self.layer['image and text decoder'](
#                 tgt=value['text embedding'], memory=value['image embedding'], 
#                 tgt_mask=value['text attention mask'], memory_mask=None, 
#                 tgt_key_padding_mask=None, memory_key_padding_mask=None
#             )
#             # print("value['image and text decoder']")
#             # print(value['image and text decoder'].shape)
#             ##  (最後預測的字, batch 這邊是 1, emb dim)
#             # print("value['image and text decoder'][-1,:,:]")
            
#             value['next word probability'] = self.layer['output'](value['image and text decoder'][-1,:,:])
#             value['next word prediction']  = value['next word probability'].argmax(axis=1).view((1,len(image)))
#             generation = torch.cat([generation, value['next word prediction']], dim=0)
#             value['output'] += [value['next word probability'].unsqueeze(axis=0)]
#             ## if(value['next word prediction'] == self.vocabulary.index['<end>']): break
#             pass
#         value['output'] = torch.cat(value['output'], axis=0)
#         # generation = generation.flatten().cpu().numpy().tolist()
#         # sentence = self.vocabulary.translate(x=generation)
#         return(value['output'])

#     pass

# class video(nn.Module):

#     def __init__(self):

#         super(video, self).__init__()
#         model = torchvision.models.video.r3d_18(True)
#         self.layer = nn.Sequential(*[i for i in model.children()][:-4], nn.AdaptiveAvgPool3d((2,16,16)))
#         return

#     def forward(self, x):

#         y = self.layer(x).permute(1,0,2,3,4).flatten(2,-1)
#         return(y)

#     pass

'''
影像模型輸入的形狀是（batch, channel, height, width），
影像模型輸出的形狀是（channel, batch, height * width），
目的是將影像模型輸出當作 word embedding 來使用，想像成（length, batch, embedding）。
'''
class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        model = torchvision.models.resnet34(True)
        layer = {}
        layer['01'] = nn.Sequential(*[i for i in model.children()][:-6])
        layer['02'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=constant['image embedding size'], nhead=4), 
            num_layers=4, norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        value = {}
        value['01'] = self.layer['01'](x).permute(1,0,2,3).flatten(2,-1)
        value['02'] = self.layer['02'](value['01'])
        y = value['02']
        # y = self.layer(x).permute(1,0,2,3).flatten(2,-1)
        return(y)

    pass
# x = torch.randn((12,3,64,64))
# image()(x).shape
'''
文字模型輸入的形狀是（length, batch），
文字模型輸出的形狀是（length, batch, embedding）。
'''
class text(nn.Module):

    def __init__(self, vocabulary=None):
    
        super(text, self).__init__()
        self.vocabulary = vocabulary
        layer = {}
        layer['01'] = nn.Embedding(self.vocabulary.size, constant['text embedding size'])
        # layer['01'] = nn.Embedding(5000, constant['text embedding size'])        
        layer['02'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=constant['text embedding size'], nhead=2), 
            num_layers=4, norm=None
        )
        # self.layer = nn.Embedding(self.vocabulary.size, self.size)
        self.layer = nn.ModuleDict(layer)

    def forward(self, x):
        
        value = {}
        # value['padding tag'] = 0
        value['mask'] = mask().sequence(x, 'autoregressive')
        value['padding mask'] = mask().padding(x, self.vocabulary.index['<padding>'])
        value['01'] = self.layer['01'](x)
        value['02'] = self.layer['02'](
            src=value['01'], 
            mask=value['mask'], 
            src_key_padding_mask=value['padding mask']
        )
        y = value['02']
        # y = self.layer(x.long()) * math.sqrt(self.size)
        return(y)

# import torch
# x = torch.randint(0,200, (17,4))
# text()(x)

def cost(skip=None):

    function = torch.nn.CrossEntropyLoss(ignore_index=skip)
    return(function)

def optimizer(model=None):
    
    if(model):
        
        function = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return(function)

    return

'''
遮罩分兩種，有 padding 以及 sequence ，
遮罩 padding 的輸入形狀是（length, batch），參數 value 輸入是 padding 符號的編碼，
假設是 1 ，就會將輸入的 tensor 中的是 1 轉換成 true ，其餘為 false ， 其中 true 代表會 mask 掉並且不參與 attention 機制，
輸出形狀是（batch, length）。
遮罩 sequence 有兩種模式，分別是 encoder 與 decoder 模式。
'''
class mask:

    def __init__(self):

        return

    def padding(self, x=None, value=None):

        # print(x)
        # print(x.shape)
        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(self, x, mode='default'):

        if(mode=='default'):

            length = len(x)
            y = torch.full((length,length), bool(False))
            y = y.cuda() if(x.is_cuda) else y.cpu()
            pass

        if(mode=='autoregressive'):

            length = len(x)
            y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
            y = y.cuda() if(x.is_cuda) else y.cpu()
            pass

        return(y)

    pass


# class mask:

#     def __init__(self, vocabulary=None):
        
#         self.vocabulary = vocabulary
#         return

#     def transform(self, x, to='attention mask'):

#         if(to=='padding mask'):

#             y = (x==self.vocabulary.index['<padding>']).transpose(0,1)
#             return(y)

#         if(to=='attention mask'):

#             length = len(x)
#             y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)

#             return(y)

#         pass

#     pass




# class PositionalEncoding(nn.Module):

#     def __init__(self):

#         super(PositionalEncoding, self).__init__()

#         emb_size = 512
#         dropout = 0.1
#         maxlen = 5000
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)
#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)
#         return        

#     def forward(self, token_embedding):

#         y = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
#         return(y)


# class TokenEmbedding(nn.Module):

#     def __init__(self, vocab_size, emb_size):
    
#         super(TokenEmbedding, self).__init__()

#         self.embedding = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size

#     def forward(self, tokens):
        
#         y = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
#         return(y)
        