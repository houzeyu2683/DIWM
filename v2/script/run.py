
import data
import network

tabulation = data.tabulation(path='../resource/Flickr/csv/index.csv')
tabulation.read()
tabulation.split(train=0.8, validation=0.1, test=0.1, target=None)

vocabulary = data.vocabulary(tabulation.table['text'])
vocabulary.build()

loader = data.loader(
    train=tabulation.train, 
    validation=tabulation.validation, 
    test=tabulation.test, 
    batch=4, vocabulary=vocabulary
)

model = network.model(vocabulary)
cost = network.cost(skip=vocabulary.index['<padding>'])
optimizer = network.optimizer(model=model)
machine = network.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log/v1-1/', checkpoint=0)

epoch = 10
for e in range(epoch):

    machine.learn(train=loader.train, validation=loader.validation)
    # machine.evaluate(train=loader.train, validation=loader.validation)
    machine.save(what='history')
    if(e==0 or e==epoch-1 or e%5==0): machine.save(what='checkpoint')
    machine.update(what='checkpoint')

    '''測試一些資料來確認模型是否真的有在學習。'''
    batch = next(iter(loader.train))
    image = batch['image'][0:1]
    index = machine.model.predict(image=image, device='cuda', limit=20)
    print(batch['item']['text'].iloc[0])
    print(vocabulary.decode(index))    
    pass

# batch = next(iter(loader.train))
# batch['item'].iloc[2]['text']
# x = batch['image'][2:3,:,:,:]
# y = machine.model.autoregressive(image=x, device='cpu', limit=20)
# machine.model.vocabulary.reverse(y)

'''mask default and regressive prediction '''