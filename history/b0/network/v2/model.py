

import torch
import torchvision
from torch import nn
from transformers import BertModel

constant = {
    'image embedding size':768,
    'text embedding size':768
}

'''
模型主要架構，其中分別由各種模組來完成。
'''
class model(nn.Module):

    def __init__(self, vocabulary=None):

        super(model, self).__init__()
        self.vocabulary = vocabulary
        # layer = {}
        # layer['01'] = cross(self.vocabulary)
        self.layer = cross(self.vocabulary)
        pass
    
    def forward(self, batch, recurse=False, device='cpu'):

        self.layer = self.layer.to(device)
        if(not recurse):

            '''
            利用 transformer 方法來產生序列，是 RNN 中的 teacher forcing 概念。
            '''
            image, text = batch['image'].to(device), batch['text'].to(device)
            value = self.layer([image, text])
            y = value
            return(y)

        else:
            
            '''
            利用遞迴的方式來產生預測的序列，是 RNN 的概念。
            '''
            image  = batch['image'].to(device)
            size   = batch['size']
            length = batch['max text length']
            generation  = torch.full((1, size), self.vocabulary.index('<start>'), dtype=torch.long).to(device)
            value = {}
            for i in range(length-1):

                if(i==0):

                    value['next probability'] = self.layer([image, generation])[-1,:,:]
                    value['next prediction']  = value['next probability'].argmax(axis=1).unsqueeze(dim=0)
                    value['output probability'] = value['next probability'].unsqueeze(0)
                    pass

                else:

                    value['next probability'] = self.layer([image, generation])[-1,:,:]
                    value['next prediction']  = value['next probability'].argmax(axis=1).unsqueeze(dim=0)
                    value['output probability'] = torch.cat([value['output probability'], value['next probability'].unsqueeze(dim=0)], dim=0)
                    
                    pass
                
                generation = torch.cat([generation, value['next prediction']], dim=0)
                pass

            y = value['output probability']
            return(y)

    '''輸入 image 維度 (1, 3, 224, 224) ，'''
    def predict(self, image=None, device='cpu', limit=20):

        self.layer = self.layer.to(device)
        image = image.to(device)
        if(len(image)!=1): return("the image dimension need (1, 3, 224, 224) shape")
        generation = torch.full((1, 1), self.vocabulary.index('<start>'), dtype=torch.long).to(device)
        value = {}
        for _ in range(limit-1):
            
            with torch.no_grad():

                value['next probability'] = self.layer([image, generation])[-1,:,:]
                value['next prediction']  = value['next probability'].argmax(axis=1).view((1,1))
                generation = torch.cat([generation, value['next prediction']], dim=0)
                if(value['next prediction'] == self.vocabulary.index('<end>')): break
                pass

            pass

        y = list(generation.view(-1).cpu().numpy())
        return(y)

    pass

'''
影像與文字做 cross attention 機制，輸入一個序列，輸出一個長度一樣的序列。
'''
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
        layer['04'] = nn.Linear(constant['text embedding size'], self.vocabulary.length)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        value = {}
        value['image'], value['text'] = x
        value['01'] = self.layer['01'](value['image'])
        value['02'] = self.layer['02'](value['text'])
        value['03'] = self.layer['03'](
            tgt=value['02'], memory=value['01'], 
            tgt_mask = mask.sequence(x=value['text'], recurse=True),
            memory_mask = None, 
            tgt_key_padding_mask = mask.padding(value['text'], self.vocabulary.index('<padding>')),
            memory_key_padding_mask = None
        )
        value['04'] = self.layer['04'](value['03'])
        y = value['04']
        return(y)

    pass


'''
針對影像做 image embedding 機制。
'''
class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        model = torchvision.models.resnet34(True)
        layer = {}
        layer['01'] = nn.Sequential(*[i for i in model.children()][:-2])
        layer['02'] = nn.Linear(49, constant['image embedding size'])
        layer['03'] = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=constant['image embedding size'], nhead=4), 
            num_layers=2, norm=None
        )
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x):

        value = {}
        value['01'] = self.layer['01'](x)
        
        '''
        為了輸入自注意力機制模塊，
        需要將輸出的維度由 (batch, channel, hieght, width) 轉成 (channel, batch, embedding) ，
        注意力模塊輸入才會是正確的。
        '''
        value['02'] = self.layer['02'](value['01'].flatten(2)).permute(1, 0, 2)
        value['03'] = self.layer['03'](value['02'])
        y = value['03']
        return(y)

    pass
# x = torch.randn((12,3,224, 224))
# image()(x).shape
'''
Pytorch transformer layer:
文字模型輸入的形狀是（length, batch），
文字模型輸出的形狀是（length, batch, embedding）。
Transformers 套件:
文字模型輸入的形狀是（batch, length），
文字模型輸出的形狀是（batch, length, embedding）。
'''

'''
最機掰的一個模組， 
transformers 套件中序列執行 embedding 的輸入維度是 (batch, length) ，輸出是 (batch, length, embedding) 。 
而 torch 中是 (length, batch) ，輸出是 (length, batch, embedding) ，
我真它媽不懂為何要不一樣。
這個模組將文字特徵做詞嵌入，使用 BERT 預訓練模型來執行。
'''
class text(nn.Module):

    def __init__(self, vocabulary=None):
    
        super(text, self).__init__()
        self.vocabulary = vocabulary
        model = BertModel.from_pretrained('bert-base-uncased')

        '''如果先前有在 BERT 提供的 tokenizer 中自行添加 token ，你要修正 embedding 的數量，否則會出錯。'''
        model.resize_token_embeddings(self.vocabulary.length)
        # layer = {}
        # layer['01'] = model
        # self.layer = nn.ModuleDict(layer)
        self.layer = model
        return

    def forward(self, x):
        
        '''輸入維度由 (length, batch) 改成 (batch, length) ，否則會出錯。'''
        x = x.transpose(0,1)
        y = self.layer(x)[0]

        '''輸出維度由 (batch, length, embedding) 改成 (length, batch, embedding) ，真的很機掰。'''
        y = y.transpose(0,1)
        return(y)

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
'''
class mask:

    '''
    遮罩 padding 的輸入形狀是（length, batch），參數 value 輸入是 padding 符號的編碼，
    假設是 1 ，就會將輸入的 tensor 中的是 1 轉換成 true ，其餘為 false ， 其中 true 代表會 mask 掉並且不參與 attention 機制，
    輸出形狀是（batch, length）。
    '''
    def padding(x=None, value=None):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x, recurse=False):

        if(not recurse):

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
        