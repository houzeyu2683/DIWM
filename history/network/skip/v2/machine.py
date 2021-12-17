

import os
import tqdm
import torch
import numpy
import pickle
import pandas
import plotly.graph_objects as go
import editdistance


class machine:

    def __init__(self, model=None, optimizer=None, cost=None, device='cpu', folder='LOG/(test)', checkpoint=0):

        self.model      = model
        self.optimizer  = optimizer
        self.cost       = cost
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass
        
        self.history = {'epoch':[], 'train loss':[], 'validation loss':[], 'train edit distance':[], 'validation edit distance':[]}
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
            value['video input'] = batch['video input'].to(self.device)
            value['text input']  = batch['text input'].to(self.device)
            value['text score']  = self.model(
                video=value['video input'], 
                text=value['text input'][:-1,:], 
                device=self.device
            )
            loss = self.cost(value["text score"].flatten(0,1), value['text input'][1:,:].flatten())
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
                value['video input'] = batch['video input'].to(self.device)
                value['text input']  = batch['text input'].to(self.device)
                value['text score']  = self.model(
                    video=value['video input'], 
                    text=value['text input'][:-1,:], 
                    device=self.device
                )
                loss = self.cost(value["text score"].flatten(0,1), value['text input'][1:,:].flatten())
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
                value['video input'] = batch['video input'].to(self.device)
                value['text input']  = batch['text input'].to(self.device)
                for i in range(batch['size']):

                    prediction = self.model.generate(video=batch['video input'][i:i+1,:,:,:,:], device=self.device, limit=128)
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
                value['video input'] = batch['video input'].to(self.device)
                value['text input']  = batch['text input'].to(self.device)
                for i in range(batch['size']):

                    prediction = self.model.generate(video=batch['video input'][i:i+1,:,:,:,:], device=self.device, limit=128)
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

            path = os.path.join(self.folder, "loss.html")
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=self.history['epoch'], y=self.history['train loss'],
                    mode='lines',
                    name='train loss',
                    line_color="blue"
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=self.history['epoch'], y=self.history['validation loss'],
                    mode='lines',
                    name='validation loss',
                    line_color="green"
                )
            )
            figure.write_html(path)
            pass

            path = os.path.join(self.folder, "edit distance.html")
            figure.add_trace(
                go.Scatter(
                    x=self.history['epoch'], y=self.history['train edit distance'],
                    mode='lines+markers',
                    name='train edit distance',
                    line_color="blue"
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=self.history['epoch'], y=self.history['validation edit distance'],
                    mode='lines+markers',
                    name='validation edit distance',
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




