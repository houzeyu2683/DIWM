

import os
import tqdm
import torch
import numpy
import pickle
import pandas
import editdistance
import plotly.graph_objects as go
import random


class machine:

    def __init__(self, model=None, optimizer=None, cost=None, device='cpu', folder='LOG', checkpoint=0):

        self.model      = model
        self.optimizer  = optimizer
        self.cost       = cost
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass
        
        # self.history = {'epoch':[], 'train loss':[], 'validation loss':[], "train edit distance":[], "validation edit distance":[]}
        self.history = {'epoch':[], 'train loss':[], 'validation loss':[]}
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        pass

    def learn(self, train=None, validation=None):

        self.history['epoch'] += [self.checkpoint]
        pass

        ################################################################################
        ##  On train.
        ################################################################################
        self.model = self.model.to(self.device)
        self.model.train()
        pass
        
        record  = {'train loss' : []}
        progress = tqdm.tqdm(train, leave=False)
        for batch in progress:

            self.model.zero_grad()
            value = {}
            value['image'] = batch['image'].to(self.device)
            value['text']  = batch['text'].to(self.device)
            value['score']  = self.model(
                image=value['image'], 
                text=value['text'][:-1,:], 
                device=self.device
            )
            loss = self.cost(value["score"].flatten(0,1), value['text'][1:,:].flatten())
            loss.backward()
            self.optimizer.step()
            value['loss'] = loss.item()
            progress.set_description("train loss : {}".format(value['loss']))
            record['train loss'] += [value['loss']]
            pass
        
        self.history['train loss'] += [numpy.array(record['train loss']).mean()]
        pass

        ################################################################################
        ##  On validation.
        ################################################################################
        self.model = self.model.to(self.device)
        self.model.eval()
        pass
        
        record  = {'validation loss' : []}
        progress = tqdm.tqdm(validation, leave=False)
        for batch in progress:

            with torch.no_grad():
                
                value = {}
                value['image'] = batch['image'].to(self.device)
                value['text']  = batch['text'].to(self.device)
                value['score']  = self.model(
                    image=value['image'], 
                    text=value['text'][:-1,:], 
                    device=self.device
                )
                loss = self.cost(value["score"].flatten(0,1), value['text'][1:,:].flatten())
                value['loss'] = loss.item()
                progress.set_description("validation loss : {}".format(value['loss']))
                record['validation loss'] += [value['loss']]
                pass
            
            pass

        self.history['validation loss'] += [numpy.array(record['validation loss']).mean()]
        return

    def evaluate(self, train=None, validation=None):

        ################################################################################
        ##  On train.
        ################################################################################
        self.model = self.model.to(self.device)
        self.model.eval()
        pass
        
        record  = {'train edit distance' : []}
        progress = tqdm.tqdm(train, leave=False)
        for batch in progress:

            with torch.no_grad():
                
                value = {}
                value['image'] = batch['image'].to(self.device)
                value['text']  = batch['text'].to(self.device)
                for i in range(batch['size']):

                    prediction = self.model.autoregressive(image=batch['image'][i:i+1,:,:,:], device=self.device, limit=20)
                    prediction = self.model.vocabulary.reverse(x=prediction)
                    target = batch['item'].iloc[i]["text"]
                    record['train edit distance'] += [editdistance.eval(prediction, target)]
                    pass

                pass
            
            pass

        self.history['train edit distance'] += [numpy.array(record['train edit distance']).mean()]
        pass
    
        ################################################################################
        ##  On validation.
        ################################################################################
        self.model = self.model.to(self.device)
        self.model.eval()
        pass
        
        record  = {'validation edit distance' : []}
        progress = tqdm.tqdm(validation, leave=False)
        for batch in progress:

            with torch.no_grad():
                
                value = {}
                value['image'] = batch['image'].to(self.device)
                value['text']  = batch['text'].to(self.device)
                for i in range(batch['size']):

                    prediction = self.model.autoregressive(image=batch['image'][i:i+1,:,:,:], device=self.device, limit=20)
                    prediction = self.model.vocabulary.reverse(x=prediction)
                    target = batch['item'].iloc[i]["text"]
                    record['validation edit distance'] += [editdistance.eval(prediction, target)]
                    pass

                pass
            
            pass

        self.history['validation edit distance'] += [numpy.array(record['validation edit distance']).mean()]
        return

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

            path = os.path.join(self.folder, "history.csv")
            pandas.DataFrame(self.history).to_csv(path, index=False)
            print("save the history of model to {}".format(path))
            pass

            path = os.path.join(self.folder, "history.html")
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=self.history['epoch'], y=self.history['train loss'],
                    mode='lines+markers',
                    name='train loss',
                    line_color="blue"
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=self.history['epoch'], y=self.history['validation loss'],
                    mode='lines+markers',
                    name='validation loss',
                    line_color="green"
                )
            )
            figure.write_html(path)
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




