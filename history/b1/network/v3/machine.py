

import os
import tqdm
import torch
import numpy
import pickle
import pandas
import plotly.graph_objects as go
import nltk
from nltk.translate.bleu_score import corpus_bleu
# s = corpus_bleu(list_of_references=[[[12,32,44,13], [12,32,22,13]]], hypotheses=[[12,32,44,13]])
# round(s)
track = {
    "epoch":[],
    'train loss':[],
    "validation loss": []
}


class machine:

    def __init__(self, model=None, optimizer=None, cost=None, device='cpu', folder='log', checkpoint=0):

        self.model      = model
        self.optimizer  = optimizer
        self.cost       = cost
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass
        
        self.track = track
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        pass

    def learn(self, train=None, validation=None):

        self.track['epoch'] += [self.checkpoint]
        pass

        ################################################################################
        ##  On train.
        ################################################################################
        self.model = self.model.to(self.device)
        self.model.train()
        pass
        
        iteration  = {'train loss':[]}
        progress = tqdm.tqdm(train, leave=False)
        for batch in progress:

            self.model.zero_grad()
            v = dict()
            v['image'] = batch['image'].to(self.device)
            v['text']  = batch['text'].to(self.device)
            v['score']  = self.model(
                x = (v['image'], v['text'][:-1,:]),
                device=self.device
            )
            loss = self.cost(v["score"].flatten(0,1), v['text'][1:,:].flatten())
            loss.backward()
            self.optimizer.step()
            iteration['train loss'] += [loss.item()]
            progress.set_description("train loss : {}".format(iteration['train loss'][-1]))
            pass
        
        self.track['train loss'] += [numpy.array(iteration['train loss']).mean()]
        pass

        ################################################################################
        ##  On validation.
        ################################################################################
        self.model = self.model.to(self.device)
        self.model.eval()
        pass
        
        iteration  = {'validation loss':[]}
        progress = tqdm.tqdm(validation, leave=False)
        for batch in progress:

            with torch.no_grad():

                v = dict()
                v['image'] = batch['image'].to(self.device)
                v['text']  = batch['text'].to(self.device)
                v['score']  = self.model(
                    x = (v['image'], v['text'][:-1,:]),
                    device=self.device
                )
                loss = self.cost(v["score"].flatten(0,1), v['text'][1:,:].flatten())
                iteration['validation loss'] += [loss.item()]
                progress.set_description("validation loss : {}".format(iteration['validation loss'][-1]))
                pass
            
            continue

        self.track['validation loss'] += [numpy.array(iteration['validation loss']).mean()]
        return

    def evaluate(self, loader, name):

        self.model = self.model.to(self.device)
        self.model.eval()
        pass
        
        progress = tqdm.tqdm(loader, leave=False)
        iteration = {"target":[], "prediction":[]}
        for index, batch in enumerate(progress, 0):
            
            batch['item'] = batch['item']
            with torch.no_grad():

                if((index%5)==0): 
                    
                    x = batch['image'][0:1,:, :, :].to(self.device)
                    prediction = self.model.predict(
                        x=x, 
                        device=self.device, 
                        limit=20
                    )
                    target = []
                    target += [batch['text'][:,0].tolist()]
                    pass
                
                else:

                    target += [batch['text'][:,0].tolist()]
                    pass

                pass
            
            collection = ((index+1)%5)==0
            if(collection): 
                
                iteration['target'] += [target]
                iteration['prediction'] +=[prediction]
                pass

            pass
        
        return(iteration)
        score = round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction']), 3)
        print('the gleu score {}'.format(score))
        return

# s = corpus_bleu(list_of_references=[[[12,32,44,13], [12,32,22,13]], [[1,2,3,5], [5,4,3,2,1]]], hypotheses=[[12,32,44,13], [1,2,3,4,5]])
# round(s,3)


    # def evaluate(self, loader, name):

    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     iteration  = {'target':[], 'prediction':[]}
    #     result = {'image':[], "text":[], "description":[]}
    #     progress = tqdm.tqdm(loader, leave=False)
    #     for _, batch in enumerate(progress, 1):
            
    #         batch['item'] = batch['item'].reset_index(drop=True)
    #         for b in range(batch['size']):

    #             with torch.no_grad():
                    
    #                 v = dict()
    #                 v['x'] = batch['image'][b:b+1,:, :, :].to(self.device)
    #                 v['prediction'] = self.model.predict(
    #                     x=v['x'], 
    #                     device=self.device, 
    #                     limit=20
    #                 )
    #                 v['description'] = self.model.vocabulary.decode(token=v['prediction'], text=True)
    #                 v['target'] = [filter(
    #                     lambda x: (x!= self.model.vocabulary.index['<padding>']), 
    #                     batch['text'][:,b].tolist()
    #                 )]
    #                 pass
                
    #             iteration['target'] += [v['target']]
    #             iteration['prediction'] += [v['prediction']]
    #             pass

    #             result['image'] += [batch['item']['image'][b]]
    #             result['text'] +=  [batch['item']['text'][b]]
    #             result['description'] +=  [v['description']]
    #             pass

    #         pass
        
    #     score = round(corpus_bleu(iteration['target'], hypotheses=iteration['prediction']),3)
    #     print('the gleu score {}'.format(score))
    #     pass

    #     location = os.path.join(self.folder, "{} result.csv".format(name))
    #     pandas.DataFrame(result).to_csv(location, index=False)
    #     return

    def save(self, what='checkpoint'):

        ##  Save the checkpoint.
        if(what=='checkpoint'):

            path = os.path.join(self.folder, "model-" + str(self.checkpoint) + ".checkpoint")
            with open(path, 'wb') as data:

                pickle.dump(self.model, data)
                pass

            print("save the weight of model to {}".format(path))
            pass

        if(what=='history'):

            history = pandas.DataFrame(self.track)
            history.to_csv(
                os.path.join(self.folder, "history.csv"), 
                index=False
            )
            pass

            # path = os.path.join(self.folder, "history.html")
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=history['epoch'], y=history['train loss'],
                    mode='lines+markers',
                    name='train loss',
                    line_color="blue"
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=history['epoch'], y=history['validation loss'],
                    mode='lines+markers',
                    name='validation loss',
                    line_color="green"
                )
            )
            figure.write_html(os.path.join(self.folder, "history.html"))
            pass

        return

    def update(self, what='checkpoint'):

        if(what=='checkpoint'): self.checkpoint = self.checkpoint + 1
        return

    def load(self, path, what='model'):

        if(what=='model'):

            with open(path, 'rb') as paper:
                
                self.model = pickle.load(paper)

            print('load model successfully')
            pass

        return
        
    pass






    # def evaluate(self, train=None, validation=None):

    #     ################################################################################
    #     ##  On train.
    #     ################################################################################
    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     record  = {'train edit distance' : []}
    #     progress = tqdm.tqdm(train, leave=False)
    #     for batch in progress:

    #         with torch.no_grad():
                
    #             value = {}
    #             value['image'] = batch['image'].to(self.device)
    #             value['text']  = batch['text'].to(self.device)
    #             for i in range(batch['size']):

    #                 prediction = self.model.autoregressive(image=batch['image'][i:i+1,:,:,:], device=self.device, limit=20)
    #                 prediction = self.model.vocabulary.reverse(x=prediction)
    #                 target = batch['item'].iloc[i]["text"]
    #                 record['train edit distance'] += [editdistance.eval(prediction, target)]
    #                 pass

    #             pass
            
    #         pass

    #     self.history['train edit distance'] += [numpy.array(record['train edit distance']).mean()]
    #     pass
    
    #     ################################################################################
    #     ##  On validation.
    #     ################################################################################
    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass
        
    #     record  = {'validation edit distance' : []}
    #     progress = tqdm.tqdm(validation, leave=False)
    #     for batch in progress:

    #         with torch.no_grad():
                
    #             value = {}
    #             value['image'] = batch['image'].to(self.device)
    #             value['text']  = batch['text'].to(self.device)
    #             for i in range(batch['size']):

    #                 prediction = self.model.autoregressive(image=batch['image'][i:i+1,:,:,:], device=self.device, limit=20)
    #                 prediction = self.model.vocabulary.reverse(x=prediction)
    #                 target = batch['item'].iloc[i]["text"]
    #                 record['validation edit distance'] += [editdistance.eval(prediction, target)]
    #                 pass

    #             pass
            
    #         pass

    #     self.history['validation edit distance'] += [numpy.array(record['validation edit distance']).mean()]
    #     return