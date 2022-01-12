

import torch
import torchvision
from torch import nn
import math 


'''
影像輸入神經網路模組，首先針對整張影像進行特徵萃取，
在輸出層之前舖平並執行注意力機制。
'''
class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        shape = (2,3,224,224)
        self.example = torch.randn(shape)
        pass

        layer = dict()
        layer['1'] = nn.Sequential(
            *[i for i in torchvision.models.resnet152(True).children()][:-4], 
            nn.Flatten(2)
        )
        layer['2'] = nn.Sequential(
            nn.Linear(4107, 784),
            nn.Tanh()
        )
        layer['3'] = nn.Sequential(nn.Linear(784, 128), nn.Tanh())
        layer['4'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=1,
            norm=None
        )
        layer['5'] = nn.Linear(128, 300)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='image'):

        v = dict()
        v['1'] = self.layer['1'](x).permute(1,0,2)
        v['2'] = self.layer['2'](self.patch(x=x, block=(6,6)))
        v['3'] = self.layer['3'](torch.cat([v['1'], v['2']], dim=0))
        v['4'] = self.layer['4'](src =v['3'], mask = None, src_key_padding_mask = None)
        v['5'] = self.layer['5'](v['4'])
        pass
        
        y = v['5']
        return(y)

    def test(self):

        y = self.forward(x=self.example)
        return(y)

    def patch(self, x='image', block=(6,6)):

        heigh, width = math.floor(x.shape[2] / block[0]), math.floor(x.shape[3] / block[1])
        p = []
        for h in range(block[0]):

            for w in range(block[1]):
                
                b = x[:, :, (h * heigh):((h + 1) * heigh), (w * width):((w + 1) * width)]
                p += [b.unsqueeze(axis=0)]
                continue

            continue
        
        p = torch.cat(p, axis=0)
        p = p.flatten(2)
        '[p] : (length, batch, embedding)'
        return(p)

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
        shape = (7, 2)
        self.example = torch.randint(0,20, shape)
        pass

        self.vocabulary = vocabulary
        size = self.vocabulary.size if(self.vocabulary) else 100
        pass

        layer = dict()
        layer['e'] = nn.Embedding(size, 300)
        layer['p'] = nn.Embedding(size, 300)
        layer['1'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=300, nhead=15),
            num_layers=1,
            norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def position(self, x=None):

        empty = torch.zeros(x.shape).type(torch.LongTensor)
        for row in range(len(empty)): empty[row,:]=row
        y = empty.cuda() if(x.is_cuda) else empty.cpu()
        return(y)

    def forward(self, x=None):

        v = dict()
        v['e'] = self.layer['e'](x)
        v['p'] = self.layer['p'](x)
        v['1'] = self.layer['1'](
            src = v['e'] + v['p'], 
            mask = mask.sequence(x=x, recourse=True), 
            src_key_padding_mask = mask.padding(x=x, value=self.vocabulary.index['<padding>'])
        )
        pass
        
        y = v['1']
        return(y)

    def test(self):

        y = self.forward(x=self.example)
        return(y)


'''
影像與文本輸入注意力機制模組，
影像編碼當作記憶力輸入編碼器。
'''
class attention(nn.Module):

    def __init__(self, vocabulary=None):

        super(attention, self).__init__()
        shape = (2,3,224,224), (7, 2)
        self.example = torch.randn(shape[0]), torch.randint(0,20, shape[1])
        pass

        self.vocabulary = vocabulary
        size = self.vocabulary.size if(self.vocabulary) else 100
        pass

        layer = dict()
        layer['image'] = image()
        layer['text']  = text(vocabulary=self.vocabulary)
        layer['1'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=300, nhead=15),
            num_layers=1,
            norm=None
        )
        layer['2'] = nn.Linear(300, size)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x=('image', 'text')):

        image, text = x
        pass

        v = dict()
        v['image'] = self.layer['image'](image)
        v['text'] = self.layer['text'](text)
        v['1'] = self.layer['1'](
            tgt=v['text'], 
            memory=v['image'], 
            tgt_mask = mask.sequence(x=text, recourse=True),
            memory_mask = None, 
            tgt_key_padding_mask = mask.padding(x=text, value=self.vocabulary.index['<padding>']),
            memory_key_padding_mask = None
        )
        v['2'] = self.layer['2'](v['1'])
        y = v['2']
        return(y)

    def test(self):

        y = self.forward(x=self.example)
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