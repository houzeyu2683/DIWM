

import nltk 
import re
import string
import itertools
from collections import Counter


def tokenize(x = 'hello word!'):

    '''
    兩個英文單字或是字母中間如果有 '-' ，那定義為一個單字，在清除標點符號的過程要忽略，其餘標點符號刪掉。 
    '''
    pattern  = "[" + re.sub("[-]", "", string.punctuation) + ']'
    word = re.sub(pattern, "", x).split()

    '''
    Stemming with porter method.
    '''
    stemming = True
    if(stemming):

        porter = nltk.stem.PorterStemmer()
        word = [porter.stem(w) for w in word]
        pass
    
    return(word)


class vocabulary:

    def __init__(self, sentence=['hello world', "how are you today?"], tokenize=tokenize):

        self.sentence = sentence
        self.tokenize = tokenize
        pass

    def build(self):

        collection = []
        for s in self.sentence:

            w = self.tokenize(s)
            collection += [w]
            pass

        collection = list(itertools.chain(*collection))
        self.count = Counter(collection)
        self.index = {'<unknown>':0, "<padding>":1, "<start>":2, "<end>":3}
        self.index.update({w:t for t, w in enumerate(self.count.keys(), len(self.index))})
        self.size = len(self.index)
        return

    def transform(self, x='how are you?', to='index'):

        if(to=='index'):
            
            i = [self.index[w] if(w in self.index.keys()) else self.index['<unknown>'] for w in self.tokenize(x)]
            i = [self.index['<start>']] + i + [self.index['<end>']]
            index = i
            pass

        return(index)

    def reverse(self, x=['22', '31', '781']):

        key   = list(self.index.keys())
        value = list(self.index.values())
        word  = [key[value.index(i)] for i in x]
        sentence = " ".join(word)
        return(sentence)

    pass



    # def convert(self, index=['22', '31', '781'], to='word'):

    #     if(to=='word'):

    #         key = list(self.index.keys())
    #         value = list(self.index.values())
    #         # y = " ".join([key[value.index(i)] for i in index])
    #         word = [key[value.index(i)] for i in index]

    #     return(word)


# index = []
# for s in sentence:

#     i = [self.index[w] if(w in self.index.keys()) else self.index['<unknown>'] for w in self.tokenize(s)]
#     i = [self.index['<classification>']] + i
#     index += [i]                
#     pass
 