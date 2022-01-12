

import pandas
import torch
from sklearn.model_selection import train_test_split


class tabulation:

    def __init__(self, path):

        self.path = path
        return

    def read(self):

        self.data  = pandas.read_csv(self.path)
        self.index = self.data['image'].unique().tolist()
        self.size  = len(self.data)
        return

    def split(self, rate="(train, validation, test)"):

        train, validation, test = slice(index=self.index, train=rate[0], validation=rate[1], test=rate[2])
        self.train = self.data.loc[[i in train for i in self.data['image']]].copy().reset_index(drop=True)
        self.validation = self.data.loc[[i in validation for i in self.data['image']]].copy().reset_index(drop=True)
        self.test = self.data.loc[[i in test for i in self.data['image']]].copy().reset_index(drop=True)
        return
    
    def convert(self, what='train', to='dataset'):

        if(to=='dataset'):

            if(what=='data'): self.data = dataset(self.data)
            if(what=='train'): self.train = dataset(self.train)
            if(what=='validation'): self.validation = dataset(self.validation)
            if(what=='test'): self.test = dataset(self.test)
            pass

        return
    
    pass


class dataset(torch.utils.data.Dataset):

    def __init__(self, table):
        
        self.table = table.reset_index(drop=True)
        pass

    def __getitem__(self, index):

        item = self.table.loc[index]
        return(item)
    
    def __len__(self):

        length = len(self.table)
        return(length)

    pass


def slice(index=None, train=0.8, validation=0.0, test=0.2):

    assert index!=None
    assert train+validation+test == 1
    i = dict()
    i['train'], i['other'] = train_test_split(
        index, 
        test_size=(validation + test), 
        random_state=0
    )
    i['validation'], i['test'] = train_test_split(
        i['other'], 
        test_size=test / (validation+test), 
        random_state=0
    )
    return(i['train'], i['validation'], i['test'])

