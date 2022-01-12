

import torch
import torchvision
from torch import nn


'''
影像輸入神經網路模組，首先針對整張影像進行特徵萃取，
在輸出層之前舖平並執行注意力機制。
'''
class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        backbone = torchvision.models.resnet152(True)
        pass
        
        layer = dict()
        layer['backbone'] = nn.Sequential(
            *[i for i in backbone.children()][:-3],
            nn.Flatten(2)
        )
        layer['attention'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=196, nhead=4),
            num_layers=1,
            norm=None
        )
        layer['memory'] = nn.Sequential(
            nn.Linear(196, 196),
            nn.Sigmoid()
        )
        layer['cell'] = nn.Sequential(
            nn.Linear(196, 196),
            nn.Sigmoid()
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='(2,3,224,224)'):

        v = dict()
        v['backbone'] = self.layer['backbone'](x).permute(1,0,2)
        v['attention'] = self.layer['attention'](v['backbone'])
        v['memory'] = self.layer['memory'](v['attention'].mean(0)).unsqueeze(0)
        v['cell'] = self.layer['cell'](v['attention'].mean(0)).unsqueeze(0)
        pass
        
        y = v['memory'], v['cell']
        return(y)

    pass


'''
文本輸入神經網路模組，位置、文字編碼，執行順序遮罩注意力機制模組。
'''
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


class text(nn.Module):

    def __init__(self, vocabulary=None):
    
        super(text, self).__init__()
        self.vocabulary = vocabulary
        layer = dict()
        layer['vector'] = nn.Embedding(self.vocabulary.size, 300)
        layer['position'] = nn.Embedding(self.vocabulary.size, 300)
        self.layer = nn.ModuleDict(layer)
        return

    def position(self, x="(length, text)"):

        empty = torch.zeros(x.shape).type(torch.LongTensor)
        for row in range(len(empty)): empty[row,:]=row
        y = empty.cuda() if(x.is_cuda) else empty.cpu()
        return(y)

    def forward(self, x="(length, text)"):

        v = dict()
        v['vector'] = self.layer['vector'](x)
        v['position'] = self.layer['position'](x)
        pass
        
        y = v['vector'] + v['position']
        return(y)


'''
影像與文本輸入注意力機制模組，
影像編碼當作記憶力輸入編碼器。
'''
class attention(nn.Module):

    def __init__(self, vocabulary=None):

        super(attention, self).__init__()
        self.vocabulary = vocabulary
        self.image = image()
        self.text = text(vocabulary=self.vocabulary)
        pass

        layer = dict()
        layer['decoder'] = nn.LSTM(
            input_size=300, 
            hidden_size=196, 
            num_layers=1
        )
        layer['output'] = nn.Linear(196, self.vocabulary.size)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x=('image', 'text')):

        image, text = x
        v = dict()
        v['memory'], v['cell'] = self.image(image)
        v['text'] = self.text(text)
        v['decoder'], _ = self.layer['decoder'](
            v['text'],
            (v['memory'], v['cell'])
        )
        v['output'] = self.layer['output'](v['decoder'])
        y = v['output']
        return(y)

    pass


'''
模型封包。
'''
class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        layer = attention(vocabulary=self.vocabulary)
        self.layer = layer
        pass
    
    def forward(self, x=("image", "text"), device='cpu'):

        image, text = x
        pass

        self.layer = self.layer.to(device)
        image = image.to(device)
        text = text.to(device)
        pass
        
        y = self.layer(x=(image, text))
        return(y)

    def predict(self, x='image', device='cpu', limit=20):

        if(len(x)!=1): return("the image dimension need (1, channel, heigh, width) shape")
        self.layer = self.layer.to(device)
        x = x.to(device)
        pass

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
        
        function = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
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