
import nltk 
import re
import string
import itertools
from collections import Counter

def tokenize(x = 'hello word!'):

    '''
    中間如果有 '-' ，定義為一個單字，在清除標點符號的過程要忽略，其餘標點符號刪掉。 
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

    def __init__(self, tokenize=tokenize):

        self.tokenize = tokenize
        pass

    def build(self, sentence=['hello world', "how are you today?"]):

        collection = []
        for s in sentence:

            w = self.tokenize(s)
            collection += [w]
            pass

        collection = list(itertools.chain(*collection))
        self.count = Counter(collection)
        self.index = {'<unknown>':0, "<padding>":1, "<start>":2, "<end>":3}
        self.index.update({w:t for t, w in enumerate(self.count.keys(), len(self.index))})
        self.size = len(self.index)
        return

    def encode(self, text='how are you?'):

        i = [self.index[w] if(w in self.index.keys()) else self.index['<unknown>'] for w in self.tokenize(text)]
        i = [self.index['<start>']] + i + [self.index['<end>']]
        token = i
        return(token)

    def decode(self, token=['22', '31', '781'], text=False):

        key   = list(self.index.keys())
        value = list(self.index.values())
        word  = [key[value.index(i)] for i in token]
        sentence = " ".join(word) if(text) else word
        return(sentence)

    pass
