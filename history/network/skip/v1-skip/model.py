
import math
import torch
import torchvision
from torch import nn

constant = {
    'text embedding size':784
}

class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        self.mask = mask(vocabulary=self.vocabulary)
        layer = {}
        layer['image embedding'] = image()
        layer['text embedding']  = text(vocabulary=self.vocabulary)
        layer['image and text decoder'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=constant['text embedding size'], 
                nhead=7, dim_feedforward=2048, dropout=0.1, activation='relu'
            ), 
            num_layers=3, norm=None
        )
        layer['output'] = nn.Linear(constant['text embedding size'], self.vocabulary.size)
        self.layer = nn.ModuleDict(layer)        
        return

    def forward(self, image=None, text=None, device='cpu'):

        image = image.to(device)
        text = text.to(device)
        self.layer = self.layer.to(device)
        pass

        value = {}
        value['image embedding'] = self.layer['image embedding'](image)
        value['text embedding'] = self.layer['text embedding'](text)
        value['text attention mask'] = self.mask.transform(text, to='attention mask').to(device)
        value['text padding mask'] = self.mask.transform(text, to='padding mask').to(device)
        value['image and text decoder'] = self.layer['image and text decoder'](
                tgt=value['text embedding'], memory=value['image embedding'], 
                tgt_mask=value['text attention mask'], memory_mask=None, 
                tgt_key_padding_mask=value['text padding mask'], memory_key_padding_mask=None
        )
        value['output'] = self.layer['output'](value['image and text decoder'])
        return(value['output'])

    def generate(self, image=None, device=None, limit=20):

        image = image.to(device)
        generation = torch.full((1, 1), self.vocabulary.index['<start>'], dtype=torch.long).to(device)
        self.layer = self.layer.to(device)
        pass

        value = {}
        value['image embedding'] = self.layer['image embedding'](image)
        
        for _ in range(limit):

            value['text embedding'] = self.layer['text embedding'](generation)
            value['text attention mask'] = self.mask.transform(x=generation, to='attention mask')
            value['image and text decoder'] = self.layer['image and text decoder'](
                tgt=value['text embedding'], memory=value['image embedding'], 
                tgt_mask=value['text attention mask'], memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None
            )
            # value['image and text decoder'] = value['image and text decoder'][-1,:,:]  
            ##  (最後預測的字, batch 這邊是 1, emb dim)
            # out = out.transpose(0, 1)
            value['next word probability'] = self.layer['output'](value['image and text decoder'][-1,:,:])

            value['next word prediction'] = value['next word probability'].argmax(axis=1).view((1,1))
            # _, next_word = torch.max(prob, dim=1)
            # next_word = next_word.item()
            if(value['next word prediction'] == self.vocabulary.index['<end>']): break
            generation = torch.cat([generation, value['next word prediction']], dim=0)
            # torch.full((1, 1), next_word).type(torch.long).to(device)# .type_as(src.data).fill_(next_word)
            # ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            pass
        
        generation = generation.flatten().numpy().tolist()
        sentence = self.vocabulary.translate(index=generation)
        return(sentence)

    pass

class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        model = torchvision.models.resnet34(True)
        self.layer = nn.Sequential(*[i for i in model.children()][:-4])
        return

    def forward(self, x):

        y = self.layer(x).permute(1,0,2,3).flatten(2,-1)
        return(y)

    pass

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

    def __init__(self, vocabulary=None):
        
        self.vocabulary = vocabulary
        return

    def transform(self, x, to='attention mask'):

        if(to=='padding mask'):

            y = (x==self.vocabulary.index['<padding>']).transpose(0,1)
            return(y)

        if(to=='attention mask'):

            length = len(x)
            y = torch.triu(torch.full((length,length), float('-inf')), diagonal=1)

            return(y)

        pass

    pass




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
        