
import data

tabulation = data.tabulation(path='./resource/flickr/csv/information.csv')
tabulation.read()
tabulation.split(rate=(0.8, 0.1, 0.1))
tabulation.convert(what='train', to='dataset')
tabulation.convert(what='validation', to='dataset')
tabulation.convert(what='test', to='dataset')

vocabulary = data.vocabulary()
vocabulary.build(sentence=tabulation.data['text'])

batch = 1
loader = data.loader(batch=batch, vocabulary=vocabulary)
loader.define(dataset=tabulation.train, name='train')
loader.define(dataset=tabulation.validation, name='validation')
loader.define(dataset=tabulation.test, name='test')

import network
 
model = network.v3.model(vocabulary=vocabulary)
cost = network.v3.cost(skip=-100)
optimizer = network.v3.optimizer(model=model)
machine = network.v3.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log(v3)/', checkpoint=0)

machine.load(path='log(v3)/model-10.checkpoint', what='model')
evaluation = dict()
evaluation['validation bleu score'] = machine.evaluate(loader=loader.validation)
evaluation['test bleu score'] = machine.evaluate(loader=loader.test)
evaluation