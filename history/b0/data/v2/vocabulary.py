
from transformers import BertTokenizer

tokenize = BertTokenizer.from_pretrained("bert-large-uncased")
_ = tokenize.add_special_tokens({'unk_token': '<unknown>', 'pad_token':"<padding>"})
_ = tokenize.add_special_tokens({'additional_special_tokens': ['<start>', '<end>']})

class vocabulary:

    def __init__(self, tokenize=tokenize):

        self.tokenize = tokenize
        self.length   = len(tokenize)
        pass

    def encode(self, x='how are you?'):

        index = self.tokenize.encode("<start> " + x + " <end>", add_special_tokens=False)
        return(index)

    def decode(self, x=[444,321,123,5151]):

        word = self.tokenize.decode(x)
        sentence = "".join(word)
        return(sentence)

    def index(self, x=None):

        index = self.tokenize.encode(x, add_special_tokens=False)[0] 
        return(index)

    pass


