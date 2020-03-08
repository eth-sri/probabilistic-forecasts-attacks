"""
Copyright 2020 The Secure, Reliable, and Intelligent Systems Lab, ETH Zurich
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np
import matplotlib.pyplot as pypl
import torch.nn as nn

import sys
sys.path.insert(1, 'common')
import utils

from model_tcn import TCNModel
from model_mdn import MDNModel
from model_lstm import LSTMModel

class Metrics:

    def __init__(self,model,loss_function,std,params=None,
                 pr=False,k=10,debug=False,predictions=None):

        self.k = k
        self.model = model
        self.pr = pr
        self.debug = debug

        self.predictions = predictions

        if(self.pr):
            print("Mean",self.model.mean)
        self.loss_function = loss_function

        self.keys = ["training loss (MSE)",
                "validation loss (MSE)",
                "std",
                "Daily return",
                "training loss (MDN)",
                "validation loss (MDN)"]

        if isinstance(self.model,MDNModel):
            self.type = "MDN"
        elif isinstance(self.model,LSTMModel):
            self.type = "LSTM"
        elif isinstance(self.model, TCNModel):
            self.type = "TCN"
        else:
            self.type = "DTCN"

        if params is not None :
            self.modify = utils.modify_return(params)

    def compute_val_loss(self):

        val_loss = self.compute_loss("validation", self.loss_function)

        return val_loss

    def compute_metrics(self):

        if self.type == "MDN" or self.type == "DTCN":
            mseLoss = utils.MSELoss()
        else:
            mseLoss = nn.MSELoss()


        daily_result = self.compute_average_daily_result()
        vals = [self.compute_loss("training",mseLoss),
                self.compute_loss("validation",mseLoss),
                self.compute_std(),
                daily_result]

        if self.type == "MDN" or self.type == "DTCN":

            vals += [
                self.compute_loss("training", self.loss_function),
                self.compute_loss("validation", self.loss_function),
            ]
            val_loss = vals[-1]
        else:
            vals += [
                0.,
                0.
            ]
            val_loss = vals[1]

        return vals,val_loss

    def initial_values(self):

        vals = self.compute_metrics()
        return vals

    def compute_losses(self,epoch):

        vals = self.compute_metrics()

        return vals


    def get_loss_on_set(self,xs,ys,loss_f):

        with torch.no_grad():

            output,_ = self.model(xs,"teacher forcing",y=ys)

            loss = loss_f(output, ys)
            return loss.item()

    def compute_loss(self,mode,loss_f):

        if(mode == "training"):
            n_its = self.model.datareader.n_train_batches
            method = self.model.datareader.get_training_batch
        elif(mode == "test"):
            n_its = self.model.datareader.n_test_batches
            method = self.model.datareader.get_test_batch
        elif(mode == "validation"):
            n_its = self.model.datareader.n_val_batches
            method = self.model.datareader.get_val_batch
        else:
            n_its = "none"


        losses = []
        if(self.debug):
            n_its = 1
        for i in range(n_its):

            # Get minibatch
            x,y = method(i)
            loss = self.get_loss_on_set(x,y,loss_f)

            losses.append(loss)

        return np.mean(np.array(losses))

    def compute_1_step_test_loss(self):

        n_its = self.model.datareader.n_test_batches
        method = self.model.datareader.get_test_batch
        loss_f = nn.MSELoss()
        losses = []
        for i in range(n_its):
            # Get minibatch
            x, y = method(i)
            with torch.no_grad():
                output, _ = self.model(x, "1 step")

                loss = loss_f(output, y[:,0])
                return loss.item()

            losses.append(loss)

        return np.mean(np.array(losses))


    # Function to compute the return of an investment strategy
    def compute_average_daily_result(self):

        with torch.no_grad():

            n_days = self.model.datareader.n_test_batches
            sum_returns = 0.
            batch_days = 10

            if(self.debug):
                n_days = 1

            for i in range(0,n_days,batch_days):

                bound = min(batch_days,n_days-i)
                x_cat = []
                y_cat = []
                for j in range(bound):
                    x,y = self.model.datareader.get_test_batch(i+j)
                    x_cat.append(x)
                    y_cat.append(y)

                n_per_day = x_cat[0].shape[1]

                x_cat = torch.cat(x_cat,dim=1).to(self.model.device)

                output_cat,_ = self.model(x_cat,mode="1 step")

                for j in range(bound):

                    left = n_per_day*j
                    right = n_per_day*(j+1)
                    if self.type == "LSTM" or self.type == "TCN":
                        output = output_cat[left:right].view(-1).cpu().numpy()
                    else:
                        # Adapted for several component
                        means = output_cat[0,left:right]
                        logits = output_cat[2,left:right]
                        expectation = means*logits

                        output = torch.sum(expectation,dim = 2)

                        # Only for one component
                        output = output.view(-1).cpu().numpy()
                    y = y_cat[j].cpu().numpy()

                    # Get indexes of k greatest values and k smallest values
                    largest_indexes = np.argpartition(output,-self.k,axis=0)[-self.k:]
                    smallest_indexes = np.argpartition(output,self.k-1,axis=0)[:self.k]

                    effective_pos_returns = y[largest_indexes,0]

                    effective_neg_returns = y[smallest_indexes,0]

                    mean = (np.mean(effective_pos_returns)-np.mean(effective_neg_returns))/2.
                    result = mean*self.model.dev
                    sum_returns += result

            av_result = sum_returns/float(n_days)

            return av_result.item()

    # Estimate prediction variance for the model, on test set
    def compute_std(self):
        with torch.no_grad():
            n_its = self.model.datareader.n_test_batches
            method = self.model.datareader.get_test_batch

            outs = []

            if self.debug:
                n_its = 1

            for i in range(n_its):
                # Get minibatch
                x, _ = method(i)
                output,_ = self.model(x,"1 step")
                output = output.cpu().numpy()
                outs.append(output)

            if self.type == "LSTM" or self.type == "TCN":
                outs = np.concatenate(outs, axis=0)
                std = np.std(outs)
            else:

                outs = np.concatenate(outs, axis=1)
                std = np.std(outs[0,:,:,0].reshape(-1))

            return std



