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
import torch.optim as optim
import pandas as pd
import os
from random import shuffle

import sys
sys.path.insert(1, 'dataset')
sys.path.insert(1, 'common')
from generation import data_reader
import utils, loop_utils
import metrics

OUTPUT_LENGTH=50

class Training():

    def __init__(self,year,training_params,folder,debug):

        model = training_params["model"]
        self.folder = folder
        self.debug = debug

        if(self.debug):
            self.max_epochs = 2
        else:
            self.max_epochs = training_params["max epochs"]

        if not os.path.exists(folder):
            os.mkdir(folder)

        # Get data
        self.device = utils.choose_device()
        self.datareader = data_reader.DataReader(year, self.device)

        self.model = loop_utils.generate_model(self.datareader, model, OUTPUT_LENGTH)

        self.model.datareader = self.datareader

        self.model.to(self.device)

        self.loss_function = utils.choose_loss(model)

        self.optimizer = optim.RMSprop(self.model.parameters(),lr=training_params["lr"])

        self.training_params = training_params

        self.number_concat_batches = training_params["batch"]



    def training_step(self,ind_list,i):

        # Clear gradients
        self.model.zero_grad()

        for l in range(min(self.number_concat_batches,self.datareader.n_train_batches-i)):
            xs, ys = self.datareader.get_training_batch(ind_list[i+l])

            # Run forward pass
            output,_ = self.model(xs,"teacher forcing",y=ys)

            # Compute gradients and optimize
            loss = self.loss_function(output, ys)

            loss.backward()

        self.optimizer.step()


    def train(self):

        # Data for logs
        values_history = []

        # Metrics
        metric = metrics.Metrics(self.model,
                                 self.loss_function,
                                 self.datareader.meta["std"],
                                 k=10,
                                 debug=self.debug)

        values,val_loss = metric.initial_values()
        values_history.append(values)

        print("Initial", values)

        # Early stopping
        best_iteration = -1
        epoch = 0
        best_val_loss = float("inf")
        best_params_file = self.folder+"model.pt"
        while(epoch - best_iteration <= self.training_params["early stopping"] and
              epoch < self.max_epochs):

            ind_list = [i for i in range(self.datareader.n_train_batches)]
            shuffle(ind_list)

            if(self.debug):
                ind_list = [0]
                self.datareader.n_train_batches = 1

            for i in range(0,self.datareader.n_train_batches,self.number_concat_batches):

                self.training_step(ind_list,i)

            if epoch%50 == 0:
                values,val_loss = metric.compute_losses(epoch)
                values_history.append(values)
                print("Epoch",epoch,values)

            else:
                val_loss = metric.compute_loss("validation",self.loss_function)


            # Update loop parameters
            if(val_loss < best_val_loss):
                best_val_loss = val_loss
                best_iteration = epoch
                torch.save(self.model, best_params_file)
            epoch+=1

        # Restore best model
        metric.model = torch.load(best_params_file)
        values, val_loss = metric.compute_losses(-1)
        values_history.append(values)

        assert(best_val_loss == val_loss)

        df = pd.DataFrame(
            data=values_history,
            columns=metric.keys)
        filename = "training_logs.csv"
        df.to_csv(self.folder+filename,index=False)

        return [best_iteration]+values


