

import torch


class mask:

    def __init__(self, vocabulary=None):
        
        self.vocabulary = vocabulary
        return

    def transform(self, x):

        y = []
        length = len(x)
        y += [torch.triu(torch.full((length,length), float('-inf')), diagonal=1)]
        y += [(x==self.vocabulary.index['<padding>']).transpose(0,1)]
        return(y)

    pass