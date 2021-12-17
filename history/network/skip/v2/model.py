
import re
import math
import torch
import torchvision
import editdistance
from torch import nn

constant = {
    'video embedding size':512,
    'text embedding size':512
}

class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        self.mask = {"decoder":mask(mode='decoder')}
        layer = {}
        layer['video embedding'] = video()
        layer['text embedding']  = text(vocabulary=self.vocabulary)
        layer['video and text decoder'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=constant['text embedding size'], 
                nhead=4, dim_feedforward=2048, dropout=0.1, activation='relu'
            ), 
            num_layers=3, norm=None
        )
        layer['output'] = nn.Linear(constant['text embedding size'], self.vocabulary.size)
        self.layer = nn.ModuleDict(layer)        
        return

    def forward(self, video=None, text=None, device='cpu'):

        video = video.to(device)
        text = text.to(device)
        self.layer = self.layer.to(device)
        pass

        value = {}
        value['video embedding'] = self.layer['video embedding'](video)
        value['text embedding'] = self.layer['text embedding'](text)
        value['text attention mask'] = self.mask['decoder'].sequence(x=text).to(device)
        value['text padding mask'] = self.mask['decoder'].padding(x=text, value=self.vocabulary.index['<padding>']).to(device)
        value['video and text decoder'] = self.layer['video and text decoder'](
                tgt=value['text embedding'], memory=value['video embedding'], 
                tgt_mask=value['text attention mask'], memory_mask=None, 
                tgt_key_padding_mask=value['text padding mask'], memory_key_padding_mask=None
        )
        value['output'] = self.layer['output'](value['video and text decoder'])
        return(value['output'])

    def generate(self, video=None, device='cpu', limit=20):

        video = video.to(device)
        generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
        self.layer = self.layer.to(device)
        pass

        value = {}
        value['video embedding'] = self.layer['video embedding'](video)
        for _ in range(limit):

            value['text embedding'] = self.layer['text embedding'](generation)
            value['text attention mask'] = self.mask['decoder'].sequence(x=generation).to(device)
            value['video and text decoder'] = self.layer['video and text decoder'](
                tgt=value['text embedding'], memory=value['video embedding'], 
                tgt_mask=value['text attention mask'], memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None
            )
            ##  (最後預測的字, batch 這邊是 1, emb dim)
            value['next word probability'] = self.layer['output'](value['video and text decoder'][-1,:,:])

            value['next word prediction'] = value['next word probability'].argmax(axis=1).view((1,1))
            generation = torch.cat([generation, value['next word prediction']], dim=0)
            if(value['next word prediction'] == self.vocabulary.index['<end>']): break
            pass
        
        generation = generation.flatten().cpu().numpy().tolist()
        sentence = self.vocabulary.translate(index=generation)
        sentence = re.sub("<end>|<start>", "", sentence)
        return(sentence)

    pass

class video(nn.Module):

    def __init__(self):

        super(video, self).__init__()
        model = torchvision.models.video.r3d_18(True)
        self.layer = nn.Sequential(*[i for i in model.children()][:-4], nn.AdaptiveAvgPool3d((2,16,16)))
        return

    def forward(self, x):

        y = self.layer(x).permute(1,0,2,3,4).flatten(2,-1)
        return(y)

    pass

# class image(nn.Module):

#     def __init__(self):

#         super(image, self).__init__()
#         model = torchvision.models.resnet34(True)
#         self.layer = nn.Sequential(*[i for i in model.children()][:-6])
#         return

#     def forward(self, x):

#         y = self.layer(x).permute(1,0,2,3).flatten(2,-1) # torch.Size([64, 4, 256])
#         return(y)

#     pass

class text(nn.Module):

    def __init__(self, vocabulary=None):
    
        super(text, self).__init__()
        self.vocabulary = vocabulary
        self.size = constant['text embedding size']
        self.layer = nn.Embedding(self.vocabulary.size, self.size)
        
    def forward(self, x):
        
        y = self.layer(x.long()) * math.sqrt(self.size)
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

    def __init__(self, mode='encoder'):

        self.mode = mode
        return

    def padding(self, x=None, value=None):

        y = (x==value).transpose(0,1)
        return(y)

    def sequence(self, x):

        if(self.mode=='encoder'):

            length = len(x)
            y = torch.full((length,length), bool(False))
            pass

        if(self.mode=='decoder'):

            length = len(x)
            y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)
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
        