

import torch
import torchvision
from torch import nn


'''
影像輸入神經網路模組。
'''
class image(nn.Module):

    def __init__(self):

        super(image, self).__init__()
        layer = dict()
        backbone = torchvision.models.resnet152(True)
        layer['1'] = nn.Sequential(
            *[i for i in backbone.children()][:-4], 
            nn.Flatten(2)
        )
        layer['2'] = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid()
        )
        layer['3'] = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid()
        )
        layer['4'] = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid()
        )
        layer['5'] = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid()
        ) 
        layer['6'] = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid()
        )
        layer['7'] = nn.Sequential(
            nn.Linear(784, 1024),
            nn.Sigmoid()
        )        
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x='(2:batch,3:channel,224:height,224:width)'):

        v = dict()
        v['1'] = self.layer['1'](x).mean(1)
        v['2'] = self.layer['2'](v['1']).unsqueeze(0)
        v['3'] = self.layer['3'](v['1']).unsqueeze(0)
        v['4'] = self.layer['4'](v['1']).unsqueeze(0)
        v['5'] = self.layer['5'](v['1']).unsqueeze(0)
        v['6'] = self.layer['6'](v['1']).unsqueeze(0)
        v['7'] = self.layer['7'](v['1']).unsqueeze(0)
        y = torch.cat([v['2'],v['3'],v['4']], 0), torch.cat([v['5'],v['6'],v['7']], 0)
        return(y)

    pass


'''
文本輸入神經網路模組。
'''
class mask:

    def padding(x="(17:length,7:token)", value="padding token value"):

        y = (x==value).transpose(0,1)
        y = y.cuda() if(x.is_cuda) else y.cpu()
        return(y)

    def sequence(x="(17:length,7:token)", recourse=False):

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
        size = self.vocabulary.size
        pass

        layer = dict()
        layer['1'] = nn.Embedding(size, 512)
        layer['2'] = nn.Embedding(size, 512)
        self.layer = nn.ModuleDict(layer)
        return

    def position(self, x="(17:length,7:token)"):

        empty = torch.zeros(x.shape).type(torch.LongTensor)
        for row in range(len(empty)): empty[row,:]=row
        y = empty.cuda() if(x.is_cuda) else empty.cpu()
        return(y)

    def forward(self, x="(17:length,7:token)"):

        v = dict()
        v['1'] = self.layer['1'](x)
        v['2'] = self.layer['2'](x)
        y = v['1'] + v['2']
        return(y)


'''
影像與文本輸入注意力機制模組，
影像編碼當作記憶力輸入編碼器。
'''
class attention(nn.Module):

    def __init__(self, vocabulary=None):

        super(attention, self).__init__()
        self.vocabulary = vocabulary
        size = self.vocabulary.size
        pass

        layer = dict()
        layer['1'] = image()
        layer['2'] = text(vocabulary=self.vocabulary)
        layer['3'] = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=3,
            dropout=0.1
        )
        layer['4'] = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=1024,nhead=4),
            num_layers=1,
            norm=None
        )
        layer['5'] = nn.Linear(1024, size)
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, x="((2,3,224,224):image, (17,7):text)"):

        image, text = x
        pass

        v = dict()
        v['1'] = self.layer['1'](image)
        v['2'] = self.layer['2'](text)
        v['3'], _ = self.layer['3'](
            v['2'],
            v['1']
        )
        h = v['1'][0]
        v['4'] = self.layer['4'](
            tgt=v['3'], 
            memory=h, 
            tgt_mask = mask.sequence(x=text, recourse=True), 
            memory_mask = None, 
            tgt_key_padding_mask = mask.padding(x=text, value=self.vocabulary.index['<padding>']),
            memory_key_padding_mask = None            
        )
        v['5'] = self.layer['5'](v['4'])
        y = v['5']
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
    
    def forward(self, x="((2,3,224,224):image, (17,7):text)", device='cpu'):

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


def cost(skip=-100):

    function = torch.nn.CrossEntropyLoss(ignore_index=skip)
    return(function)

def optimizer(model=None):
    
    if(model):
        
        function = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        return(function)

    return




