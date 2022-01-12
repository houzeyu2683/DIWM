

import pandas
import torch
import PIL.Image
import numpy
import os
from torch.utils.data import DataLoader
from torch.nn.utils import rnn
from torchvision import transforms as kit
from functools import partial


constant = {
    'image folder' : "../resource/211212/jpg/",
    'image size'   : (224, 224)
}


class loader:

    def __init__(self, train=None, validation=None, test=None, batch=32, vocabulary=None):

        ##  Vocabulary.
        self.vocabulary = vocabulary

        ##  Train loader.
        self.train = DataLoader(
            dataset=train, batch_size=batch, 
            shuffle=True , drop_last=True, 
            collate_fn=partial(self.collect, mode='train')
        ) if(train) else None
        
        ##  Validation loader.
        self.validation = DataLoader(
            dataset=validation, batch_size=batch, 
            shuffle=False , drop_last=False, 
            collate_fn=partial(self.collect, mode='validation')
        ) if(validation) else None

        ##  Test loader.
        self.test = DataLoader(
            dataset=test, batch_size=batch, 
            shuffle=False , drop_last=False, 
            collate_fn=partial(self.collect, mode='test')
        ) if(test) else None
        pass

    def collect(self, group, mode):

        padding = self.vocabulary.tokenize.encode('<padding>', add_special_tokens=False)[0]        
        batch = {
            'size'  : None, #len(group),
            'max text length':None,
            'item'  : [],
            'image' : [],
            'text'  : []
        }
        for item in group:
            
            engine = process(item=item, vocabulary=self.vocabulary, mode=mode)
            batch['item'] += [engine.item]
            batch['image'] += [engine.image()]
            batch['text'] += [engine.text()]
            pass

        batch['item']  = pandas.concat(batch['item'],axis=1).transpose()
        batch['image'] = torch.stack(batch['image'], 0)
        batch['text']  = rnn.pad_sequence(batch['text'], batch_first=False, padding_value=padding)
        batch['size']  = len(group)
        batch['max text length'] = len(batch['text'])
        return(batch)
    
    pass


class process:

    def __init__(self, item=None, vocabulary=None, mode='train'):

        self.item = item
        self.vocabulary = vocabulary
        self.mode = mode
        pass
    
    def image(self):

        path = os.path.join(constant['image folder'], self.item['image'])
        picture = PIL.Image.open(path).convert("RGB")
        picture = zoom(picture, 256)
        if(self.mode=='train'):
            
            blueprint = [
                kit.RandomHorizontalFlip(p=0.5),
                kit.RandomRotation((-45, +45)),
                # kit.ColorJitter(brightness=(0,1), contrast=(0,1),saturation=(0,1), hue=(-0.5, 0.5)),
                kit.RandomCrop(constant['image size']),
                kit.ToTensor(),
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture)
            pass

        else:
            
            blueprint = [
                kit.CenterCrop(constant['image size']),
                kit.ToTensor(),
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture)                    
            pass
            
        picture = picture.type(torch.float32)
        return(picture)

    def text(self):

        index = self.vocabulary.encode(self.item['text'])
        index = torch.tensor(index, dtype=torch.int64)
        return(index)

    pass

'''
根據圖片最短邊進行縮放，使得最短邊縮放後超過一定的長度。
'''
def zoom(picture=None, boundary=None):

    size = picture.size
    scale = (boundary // size[numpy.argmin(size, axis=None)]) + 1
    height, width = size[0]*scale, size[1]*scale
    picture = picture.resize((height, width))
    return(picture)
# import numpy
# size = (49, 119)
# boundary=76
# scale = (boundary // size[numpy.argmin(size, axis=None)]) + 1
# '''
# 載入 video 資料，輸出格式是 list ，裡面的每一個 frame 格式是 pillow 影像格式。
# '''
# def load(path, what='video'):

#     if(what=='video'):

#         video = []
#         animation = PIL.Image.open(path)
#         for l in range(animation.n_frames):

#             animation.seek(l)
#             video += [animation.convert("RGB")]
#             pass
        
#         return(video)
    
#     pass


        #     batch['video input'] += [engine.video()]
        # batch['video input'] = rnn.pad_sequence(batch['video input'], batch_first=True, padding_value=0).transpose(1,2)

    # 'video folder':"../resource/sample/gif/",
    # 'video size':(64, 64)


    # def video(self):

    #     frame = []
    #     path = os.path.join(constant['video folder'], self.item['video'])
    #     for p in load(path, what='video'):

    #         p = zoom(p, 76)
    #         if(self.mode=='train'):
                
    #             blueprint = [
    #                 kit.RandomHorizontalFlip(p=0.5),
    #                 kit.RandomRotation((-45, +45)),
    #                 kit.ColorJitter(brightness=(0,1), contrast=(0,1),saturation=(0,1), hue=(-0.5, 0.5)),
    #                 kit.RandomCrop(constant['video size']),
    #                 kit.ToTensor()
    #             ]
    #             convert = kit.Compose(blueprint)
    #             p = convert(p)
    #             pass

    #         else:

    #             blueprint = [
    #                 kit.CenterCrop(constant['video size']),
    #                 kit.ToTensor()
    #             ]
    #             convert = kit.Compose(blueprint)
    #             p = convert(p) 
    #             pass
            
    #         p = torch.unsqueeze(p, 0)
    #         frame += [p]
    #         pass

    #     frame = torch.cat(frame, 0).type(torch.float32)
    #     return(frame)

# path = "../resource/sample/gif/tumblr_l876j3kjpF1qcw5xjo1_250.gif"
# animation = PIL.Image.open(path)
# animation.is_animated


# load(path = '../resource/sample/gif/tumblr_llawpjbYle1qewsdfo1_500.gif')
    # def video(self):

    #     animation = PIL.Image.open(os.path.join(constant['video folder'], self.item['video']))
    #     self.status = animation.is_animated
    #     if(not self.status):
            
    #         print("video error")
    #         print(self.item)
    #         print("-"*50)
    #         pass

    #     else:

    #         self.channel = (3,)
    #         self.size    = (224, 224)
    #         self.length  = animation.n_frames
    #         shape = (self.length, ) + self.channel + self.size
    #         frame = numpy.empty(shape=shape)
    #         for i in range(self.length):

    #             animation.seek(i)
    #             picture = animation.convert("RGB")
    #             shape   = self.channel + self.size
    #             if(self.mode=='train'):

    #                 blueprint = [
    #                     kit.Resize(self.size), 
    #                     kit.ToTensor(),
    #                     # kit.Lambda(lambda x: 2 * x - 1)
    #                 ]
    #                 convert = kit.Compose(blueprint)
    #                 picture = convert(picture)
    #                 pass

    #             else:

    #                 blueprint = [
    #                     kit.Resize(self.size), 
    #                     kit.ToTensor(),
    #                     # kit.Lambda(lambda x: 2 * x - 1)
    #                 ]
    #                 convert = kit.Compose(blueprint)
    #                 picture = convert(picture)                    
    #                 pass

    #             frame[i,:,:,:] = picture
    #             pass
            
    #         frame = torch.tensor(frame, dtype=torch.float32)
    #         pass

    #     return(frame)



    # pass