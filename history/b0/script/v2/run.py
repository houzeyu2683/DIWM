
import data
import network

tabulation = data.v2.tabulation(path='../resource/211212/csv/index.csv')
tabulation.read()
tabulation.split(train=0.8, validation=0.1, test=0.1, target=None)

vocabulary = data.v2.vocabulary()

loader = data.v2.loader(
    train=tabulation.train, 
    validation=tabulation.validation, 
    test=tabulation.test, 
    batch=16, vocabulary=vocabulary
)

model = network.v2.model(vocabulary)
cost = network.v2.cost(skip=vocabulary.index('<padding>'))
optimizer = network.v2.optimizer(model=model)
machine = network.v2.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log/v2/', checkpoint=0)
epoch = 50
for e in range(epoch):

    machine.learn(train=loader.train, validation=loader.validation)
    
    machine.save(what='history')
    if(e==0 or e==epoch-1 or e%1==0): machine.save(what='checkpoint')
    machine.update(what='checkpoint')

    '''測試一些資料來確認模型是否真的有在學習。'''
    batch = next(iter(loader.train))
    image = batch['image'][0:1]
    index = machine.model.predict(image=image, device='cpu', limit=20)
    print(batch['item']['text'].iloc[0])
    print(vocabulary.decode(index))
    pass



# batch = next(iter(loader.train))
# batch['item'].iloc[2]['text']
# x = batch['image'][2:3,:,:,:]
# y = machine.model.autoregressive(image=x, device='cpu', limit=20)
# machine.model.vocabulary.reverse(y)

'''mask default and regressive prediction '''







# import data
# import network

# tabulation = data.v2.tabulation(path='../resource/211212/csv/index.csv')
# tabulation.read()
# tabulation.split(train=0.8, validation=0.1, test=0.1, target=None)

# vocabulary = data.v2.vocabulary()

# loader = data.v2.loader(
#     train=tabulation.train, 
#     validation=tabulation.validation, 
#     test=tabulation.test, 
#     batch=4, vocabulary=vocabulary
# )

# # len(vocabulary.tokenize)
# model = network.v2.model(vocabulary)
# cost = network.v2.cost(skip=vocabulary.index('<padding>'))
# optimizer = network.v2.optimizer(model=model)

# # batch = next(iter(loader.train))
# # x = model(image=batch['image'], text=batch['text'], device='cpu')

# machine = network.v2.machine(model=model, optimizer=optimizer, cost=cost, device='cuda', folder='log/v2-1/', checkpoint=0)

# # machine.model(batch=batch, recurse=True, device='cpu').shape
# # x = machine.model.predict(image=batch['image'][0:1,:,:,:], limit=20, device='cuda')

# epoch = 10
# for e in range(epoch):

#     machine.learn(train=loader.train, validation=loader.validation)
    
#     machine.save(what='history')
#     if(e==0 or e==epoch-1 or e%5==0): machine.save(what='checkpoint')
#     machine.update(what='checkpoint')
#     pass

# # batch = next(iter(loader.train))
# # batch['item'].iloc[2]['text']
# # x = batch['image'][2:3,:,:,:]
# # y = machine.model.autoregressive(image=x, device='cpu', limit=20)
# # machine.model.vocabulary.reverse(y)

# '''mask default and regressive prediction '''