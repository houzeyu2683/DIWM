

import pandas
import torch
from sklearn.model_selection import train_test_split


class tabulation:

    def __init__(self, path):

        self.path = path
        return

    def read(self):

        self.data = pandas.read_csv(self.path)#.iloc[range(100),:]#.sample(400, random_state=0).reset_index(drop=True)
        # self.table = pandas.read_csv(self.path).sample(100).reset_index(drop=True)
        return

    def split(self, validation=0.2):

        train, validation = train_test_split(
            self.data['image'].unique(), 
            test_size=validation, 
            random_state=0
        )
        train = [i in train for i in self.data['image']]
        validation = [i in validation for i in self.data['image']]
        self.train = self.data.loc[train].copy().reset_index(drop=True)
        self.validation = self.data.loc[validation].copy().reset_index(drop=True)
        # train, validation = train_test_split(self.data, test_size=validation, random_state=0)
        # self.train  = train.copy().reset_index(drop=True)
        # self.validation   = validation.copy().reset_index(drop=True)
        pass    
    
    def convert(self, what='train', to='dataset'):

        if(to=='dataset'):

            if(what=='data'): self.data = dataset(self.data)
            if(what=='train'): self.train = dataset(self.train)
            if(what=='validation'): self.validation = dataset(self.validation)
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



    # def split(self, train=0.8, validation=0.2, test=0, target=None):

    #     cache = {}
    #     cache['table'] = self.table.copy()
    #     cache['table']['<index>'] = range(len(cache['table']))
    #     pass

    #     ##  Split train from table.
    #     train = train / (train+validation+test)
    #     cache['train'], _ = train_test_split(
    #         cache['table'], train_size=train, random_state=0, shuffle=True, stratify=target
    #     )
    #     cache['table'] = cache['table'].loc[[i not in cache['train']['<index>'] for i in cache['table']['<index>']]].copy()
    #     pass

    #     ##  Split validation from table.
    #     validation = validation / (validation+test)
    #     cache['validation'], _ = train_test_split(
    #         cache['table'], train_size=validation, random_state=0, shuffle=True, stratify=target
    #     )
    #     cache['table'] = cache['table'].loc[[i not in cache['validation']['<index>'] for i in cache['table']['<index>']]].copy()
    #     pass

    #     ##  Split test from table.
    #     cache['test'] = cache['table']
    #     cache['table'] = cache['table'].loc[[i not in cache['test']['<index>'] for i in cache['table']['<index>']]].copy()
    #     pass

    #     self.train      = dataset(cache['train'])
    #     self.validation = dataset(cache['validation'])
    #     self.test       = dataset(cache['test'])
    #     return

    # pass