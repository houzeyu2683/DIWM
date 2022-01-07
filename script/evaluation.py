
import data

tabulation = data.tabulation(path='./resource/flickr/csv/information.csv')
tabulation.read()
tabulation.split(validation=0.2)
tabulation.convert(what='train', to='dataset')
tabulation.convert(what='validation', to='dataset')

vocabulary = data.vocabulary()
vocabulary.build(sentence=tabulation.data['text'])

loader = data.loader(batch=2, vocabulary=vocabulary)
loader.define(dataset=tabulation.train, name='train')
loader.define(dataset=tabulation.validation, name='validation')

import network
 
model = network.v1.model(vocabulary=vocabulary)
cost = network.v1.cost(skip=vocabulary.index['<padding>'])
optimizer = network.v1.optimizer(model=model)
machine = network.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log(v1)/', checkpoint=0)
machine.load(path='log(v1)/model-5.checkpoint', what='model')

machine.evaluate(loader=loader.validation, name='validation')




# machine.evaluation['target'][0]
# machine.evaluation['prediction'][0]


