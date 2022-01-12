
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

class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        layer = cross(self.vocabulary)
        self.layer = layer
        pass
    
    def forward(self, image=None, text=None, device='cpu'):

        self.layer = self.layer.to(device)
        text = text.to(device)
        image = image.to(device)
        y = self.layer(text=text, image=image)
        return(y)

    def predict(self, image=None, device='cpu', limit=20):

        self.layer = self.layer.to(device)
        image = image.to(device)
        if(len(image)!=1): return("the image dimension need (1, 3, 224, 224) shape")
        generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
        value = {}
        for _ in range(limit-1):
            
            with torch.no_grad():

                value['next probability'] = self.layer(image=image, text=generation)[-1,:,:]
                value['next prediction']  = value['next probability'].argmax(axis=1).view((1,1))
                generation = torch.cat([generation, value['next prediction']], dim=0)
                if(value['next prediction'] == self.vocabulary.index['<end>']): break
                pass

            pass

        y = list(generation.view(-1).cpu().numpy())
        return(y)
    
    pass
#     def autoregressive(self, image=None, device='cpu', limit=20):

#         self.layer = self.layer.to(device)
#         image = image.to(device)
#         generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)        
#         # generation = torch.full((1, len(image)), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
#         # value['output'] = []
#         # value['image embedding'] = self.layer['image embedding'](image)
#         # print("value['image embedding']")
#         # print(value['image embedding'].shape)
#         value = {}
#         for _ in range(limit-1):
            
#             with torch.no_grad():

#                 value['next word probability'] = self.layer['01']([image, generation])
#                 value['next word probability'] = value['next word probability'][-1,:,:]
#                 value['next word prediction']  = value['next word probability'].argmax(axis=1).view((1,1))
#                 generation = torch.cat([generation, value['next word prediction']], dim=0)
#                 if(value['next word prediction'] == self.vocabulary.index['<end>']): break
#                 pass

#             pass

#         y = list(generation.view(-1).cpu().numpy())
#         return(y)

#     pass

# '''
# import torch
# e = torch.randn((12, 256))
# m = torch.randn((512, 12, 256))
# '''

class cross(nn.Module):

    def __init__(self, vocabulary=None):

        super(cross, self).__init__()
        self.vocabulary = vocabulary
        layer = {}
        layer['image'] = image()
        layer['text']  = text(vocabulary=self.vocabulary)
        layer['01'] = nn.Sequential(nn.Linear(512, 256), nn.Tanh())
        layer['02'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=256, nhead=4),
            num_layers=4,
            norm=None
        )
        layer['03'] = nn.Linear(256, self.vocabulary.size)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, text=None, image=None):

        value = {}
        value['image embedding'], value['image memory'] = self.layer['image'](image)
        value['text embedding']  = self.layer['text'](text)
        value['cross embedding'] = match(value['text embedding'], value['image embedding'])
        value['01'] = self.layer['01'](value['cross embedding'])
        value['02'] = self.layer['02'](
            tgt=value['01'], memory=value['image memory'], 
            tgt_mask = mask.sequence(text, True), memory_mask = None, 
            tgt_key_padding_mask = mask.padding(text, self.vocabulary.index['<padding>']), 
            memory_key_padding_mask = None
        )
        value['03'] = self.layer['03'](value['02'])
        y = value['03']
        return(y)

    pass

class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        model = torchvision.models.resnet34(True)
        layer = {}
        layer['01'] = nn.Sequential(*[i for i in model.children()][:-1], nn.Flatten(), nn.Linear(512, constant['image embedding size']))
        layer['02'] = nn.Sequential(*[i for i in layer['01'].children()][:-3], nn.Flatten(2), nn.Linear(49, constant['image embedding size']))
        layer['03'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=256, nhead=4),
            num_layers=2,
            norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x=torch.randn((12, 3, 224, 224))):

        value = {}
        value['01'] = self.layer['01'](x)
        value['02'] = self.layer['02'](x).permute(1,0,2)
        value['03'] = self.layer['03'](value['02'])
        embedding, memory = value['01'], value['03']
        return(embedding, memory)

    pass

class text(nn.Module):

    def __init__(self, vocabulary=None):
    
        super(text, self).__init__()
        self.vocabulary = vocabulary
        layer = {}
        layer['01'] = nn.Embedding(self.vocabulary.size, constant['text embedding size'])
        layer['02'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=constant['text embedding size'], nhead=2), 
            num_layers=4, norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):
        
        value = {}
        value['mask'] = mask.sequence(x, True), mask.padding(x, self.vocabulary.index['<padding>'])
        value['01'] = self.layer['01'](x)
        value['02'] = self.layer['02'](
            src=value['01'], 
            mask=value['mask'][0], 
            src_key_padding_mask=value['mask'][1]
        )
        y = value['02']
        return(y)

def cost(skip=None):

    function = torch.nn.CrossEntropyLoss(ignore_index=skip)
    return(function)

def optimizer(model=None):
    
    if(model):
        
        function = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return(function)

    return

class mask:

    def padding(x=None, value=None):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x, recourse=False):

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

def match(x, y):

    z = torch.full(x.shape, 0)
    z = z.cuda() if(x.is_cuda) else z.cpu()
    length = len(x)
    for i in range(length):

        z[i, :, :] = y
        pass
    
    output = torch.cat([x, z], 2)
    return(output)

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
        
'''
<image class>
import torch
model = torchvision.models.resnet34(True)
layer = {}
layer['01'] = nn.Sequential(*[i for i in model.children()][:-1], nn.Flatten())
layer['02'] = nn.Sequential(*[i for i in layer['01'].children()][:-2], nn.Flatten(2), nn.Linear(49, 256))
layer['03'] = nn.TransformerEncoder(
    encoder_layer=nn.TransformerEncoderLayer(d_model=256, nhead=4),
    num_layers=2,
    norm=None
)
x = torch.randn((12, 3, 224, 224))
value = {}
value['01'] = layer['01'](x)
value['02'] = layer['02'](x).permute(1,0,2)
value['03'] = layer['03'](value['02'])
embedding, memory = value['01'], value['03']
memory.shape
'''