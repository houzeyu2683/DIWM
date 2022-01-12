
import data
import network

tabulation = data.v1.tabulation(path='../resource/211212/csv/index.csv')
tabulation.read()
tabulation.split(train=0.8, validation=0.1, test=0.1, target=None)

vocabulary = data.v1.vocabulary(tabulation.table['text'])
vocabulary.build()

loader = data.v1.loader(
    train=tabulation.train, 
    validation=tabulation.validation, 
    test=tabulation.test, 
    batch=128, vocabulary=vocabulary
)

model = network.v1.model(vocabulary)
cost = network.v1.cost(skip=vocabulary.index['<padding>'])
optimizer = network.v1.optimizer(model=model)
machine = network.v1.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log/v1-1/', checkpoint=0)

epoch = 40
for e in range(epoch):

    machine.learn(train=loader.train, validation=loader.validation)
    # machine.evaluate(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    if(e==0 or e==epoch-1 or e%5==0): machine.save(what='checkpoint')
    machine.update(what='checkpoint')
    pass

# batch = next(iter(loader.train))
# batch['item'].iloc[2]['text']
# x = batch['image'][2:3,:,:,:]
# y = machine.model.autoregressive(image=x, device='cpu', limit=20)
# machine.model.vocabulary.reverse(y)

'''mask default and regressive prediction '''