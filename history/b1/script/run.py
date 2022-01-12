
import data

tabulation = data.tabulation(path='./resource/flickr/csv/information.csv')
tabulation.read()
tabulation.split(validation=0.2)

tabulation.convert(what='train', to='dataset')
tabulation.convert(what='validation', to='dataset')
next(iter(tabulation.train))

vocabulary = data.vocabulary()
vocabulary.build(sentence=tabulation.data['text'])

loader = data.loader(batch=16, vocabulary=vocabulary)
loader.define(dataset=tabulation.train, name='train')
loader.define(dataset=tabulation.validation, name='validation')
next(iter(loader.train))

import network
 
model = network.v4.model(vocabulary=vocabulary)
x = next(iter(loader.train))['image'], next(iter(loader.train))['text']
model.forward(x=x).shape

cost = network.v4.cost(skip=-100)
optimizer = network.v4.optimizer(model=model)
machine = network.v4.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log(v4)/', checkpoint=0)

epoch = 20
for e in range(epoch):

    machine.learn(train=loader.train, validation=loader.validation)
    # machine.evaluate(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    if(e==0 or e==epoch-1 or e%1==0): machine.save(what='checkpoint')
    machine.update(what='checkpoint')
    pass


