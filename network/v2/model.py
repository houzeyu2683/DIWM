

import torch
import torchvision
from torch import nn
import math 


##  影像輸入神經網路模組。
class picture(nn.Module):

    def __init__(self):

        super(picture, self).__init__()
        layer = dict()
        backbone = torchvision.models.resnet101(True)
        ##  (batch, 1024, 14, 14)
        layer['01'] = nn.Sequential(*[i for i in backbone.children()][:-3])
        layer['02'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=14*14, nhead=4),
            num_layers=2,
            norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x=None):

        v = dict()
        v['01'] = self.layer['01'](x).flatten(2).permute(1,0,2)
        v['02'] = self.layer['02'](v['01'])
        y = v['02']
        return(y)

    def example(self):
        
        x = torch.randn((2, 3, 224, 224))
        return(x)

    def test(self):

        x = self.example()
        y = self.forward(x=x)
        print(y.shape)
        return

    pass


##  文本輸入神經網路模組。
class description(nn.Module):

    def __init__(self, vocabulary=None):
    
        super(description, self).__init__()
        self.vocabulary = vocabulary
        layer = dict()
        layer['01'] = nn.Embedding(self.vocabulary.size, 14*14)
        layer['02'] = nn.Embedding(self.vocabulary.size, 14*14)
        layer['03'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=14*14, nhead=14), 
            num_layers=2, 
            norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def position(self, x=None):

        i = torch.zeros(x.shape).type(torch.LongTensor)
        for k in range(len(i)): i[k,:]=k
        i = i.cuda() if(x.is_cuda) else i.cpu()
        return(i)

    def forward(self, x=None):

        v = dict()
        v['01'] = self.layer['01'](x)
        v['02'] = self.layer['02'](self.position(x=x))
        v['03'] = self.layer['03'](
            src = v['01'] + v['02'], 
            mask = mask.sequence(x=x, recourse=True), 
            src_key_padding_mask = mask.padding(x=x, value=self.vocabulary.index['<padding>'])            
        )
        y = v['03']        
        return(y)

    def example(self):

        y = torch.randint(0,20,(7,2))
        return(y)

    def test(self):

        x = self.example()
        y = self.forward(x)
        print(y.shape)
        return


##  影像與文本輸入注意力機制模組。
class attention(nn.Module):

    def __init__(self, vocabulary=None):

        super(attention, self).__init__()
        self.vocabulary = vocabulary
        layer = dict()
        layer['01'] = picture()
        layer['02'] = description(vocabulary=self.vocabulary)
        layer['03'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=14*14, nhead=14), 
            num_layers=4, 
            norm=None
        )
        layer['04'] = nn.Linear(14*14, self.vocabulary.size)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x=(None, None)):

        picture, description = x
        v = dict()
        v['01'] = self.layer['01'](picture)
        v['02'] = self.layer['02'](description)
        v['03'] = self.layer['03'](
            tgt=v['02'], 
            memory=v['01'], 
            tgt_mask=mask.sequence(x=description, recourse=True), 
            memory_mask=None, 
            tgt_key_padding_mask=mask.padding(x=description, value=self.vocabulary.index['<padding>']), 
            memory_key_padding_mask=None
        )
        v['04'] = self.layer['04'](v['03'])
        y = v['04']
        return(y)

    pass


class mask:

    def padding(x=("length", "token"), value="padding token value"):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x=("length", "token"), recourse=False):

        if(not recourse):

            length = len(x)
            y = torch.full((length,length), bool(False))
            y = y.cuda() if(x.is_cuda) else y.cpu()
            pass

        else:

            length = len(x)
            y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
            y = y.cuda() if(x.is_cuda) else y.cpu()
            pass

        return(y)

    pass


class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        layer = attention(vocabulary=self.vocabulary)
        self.layer = layer
        pass
    
    def forward(self, x=("image", "text"), device='cpu'):

        picture, description = x
        pass

        self.layer = self.layer.to(device)
        picture = picture.to(device)
        description = description.to(device)
        pass
        
        y = self.layer(x=(picture, description))
        return(y)

    def predict(self, x='image', device='cpu', limit=20):

        self.layer = self.layer.to(device)
        x = x.to(device)
        if(len(x)!=1): return("the image dimension need (1, channel, heigh, width) shape")
        generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
        v = dict()
        for _ in range(limit-1):
            
            with torch.no_grad():

                v['next probability'] = self.layer(x=(x, generation))[-1,:,:]
                v['next prediction']  = v['next probability'].argmax(axis=1).view((1,1))
                generation = torch.cat([generation, v['next prediction']], dim=0)
                if(v['next prediction'] == self.vocabulary.index['<end>']): break
                pass

            pass

        y = list(generation.view(-1).cpu().numpy())
        return(y)
    
    pass


def cost(skip=None):

    function = torch.nn.CrossEntropyLoss(ignore_index=skip)
    return(function)

def optimizer(model=None):
    
    if(model):
        
        function = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return(function)

    return








# def match(x, y):

#     z = torch.full(x.shape, 0)
#     z = z.cuda() if(x.is_cuda) else z.cpu()
#     length = len(x)
#     for i in range(length):

#         z[i, :, :] = y
#         pass
    
#     output = torch.cat([x, z], 2)
#     return(output)

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
        
# '''
# <image class>
# import torch
# model = torchvision.models.resnet34(True)
# layer = {}
# layer['01'] = nn.Sequential(*[i for i in model.children()][:-1], nn.Flatten())
# layer['02'] = nn.Sequential(*[i for i in layer['01'].children()][:-2], nn.Flatten(2), nn.Linear(49, 256))
# layer['03'] = nn.TransformerEncoder(
#     encoder_layer=nn.TransformerEncoderLayer(d_model=256, nhead=4),
#     num_layers=2,
#     norm=None
# )
# x = torch.randn((12, 3, 224, 224))
# value = {}
# value['01'] = layer['01'](x)
# value['02'] = layer['02'](x).permute(1,0,2)
# value['03'] = layer['03'](value['02'])
# embedding, memory = value['01'], value['03']
# memory.shape
# '''