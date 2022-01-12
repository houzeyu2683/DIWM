
import data

tabulation = data.tabulation(path='./resource/flickr/csv/information.csv')
tabulation.read()
tabulation.split(validation=0.2)
tabulation.convert(what='train', to='dataset')
tabulation.convert(what='validation', to='dataset')

vocabulary = data.vocabulary()
vocabulary.build(sentence=tabulation.data['text'])

loader = data.loader(batch=1, vocabulary=vocabulary)
loader.define(dataset=tabulation.train, name='train')
# loader.define(dataset=tabulation.validation, name='validation')
loader.define(dataset=tabulation.train, name='validation')

import network
 
model = network.v4.model(vocabulary=vocabulary)
cost = network.v4.cost(skip=-100)
optimizer = network.v4.optimizer(model=model)
machine = network.v4.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log(v4)/', checkpoint=0)

machine.load(path='log(v4)/model-4.checkpoint', what='model')
iteration = machine.evaluate(loader=loader.validation, name='validation')

from nltk.translate.bleu_score import corpus_bleu
# len(iteration['target'][])
# len(iteration['prediction'])
s = corpus_bleu(iteration['target'], iteration['prediction'])
round(s,5)
