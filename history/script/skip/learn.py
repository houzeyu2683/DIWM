

import data
import network


tabulation = data.tabulation(path='../resource/sample/csv/index.csv')
tabulation.read()
tabulation.split(train=0.8, validation=0.1, test=0.1, target=None)
tabulation.train


vocabulary = data.vocabulary(tabulation.table['text'])
vocabulary.build()


loader = data.loader(
    train=tabulation.train, 
    validation=tabulation.validation, 
    test=tabulation.test, 
    batch=9, vocabulary=vocabulary
)


model = network.v2.model(vocabulary)
cost = network.v2.cost(skip=vocabulary.index['<padding>'])
optimizer = network.v2.optimizer(model=model)
machine = network.v2.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log/211211/', checkpoint=0)


epoch = 20
for e in range(epoch):

    machine.learn(train=loader.train, validation=loader.validation)
    machine.evaluate(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    if(e==0 or e==epoch-1 or e%5==0): machine.save(what='checkpoint')
    machine.update(what='checkpoint')
    pass

