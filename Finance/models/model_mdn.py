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
import torch.nn as nn
import numpy as np
import time

import pyro

import sys
sys.path.insert(1, 'common')
from sampling_utils import sample_from_distrib, sample_from_distrib_reparametrized, sample_from_joint_distrib
from utils import Loss

import lstm_based_model


class MDNModel(lstm_based_model.LstmBasedModel):
    def __init__(self,mean_return,dev_return,model_config,n_steps,n_samples=0):
        super(MDNModel, self).__init__(mean_return,dev_return)

        # Number of mixture components
        self.K = model_config["n components"]
        self.n_steps = n_steps
        self.n_samples = n_samples

        # MDN output layers
        self.output_layer_loc = nn.Linear(self.hidden_dim, self.K)
        self.output_layer_scale = nn.Linear(self.hidden_dim, self.K)
        self.output_layer_logit = nn.Linear(self.hidden_dim, self.K)

    # Forward method
    # Args: inout sequence, mode
    # Returns output sequence, and target
    def forward(self,
                input,
                mode,
                n_bins=None,
                y=None,
                n_steps=None,
                predictions=None,
                binary_predictions=None,
                binning=False):

        # Preprocess input
        returns = self.preprocessing(input)

        if(mode == "teacher forcing"):

            outputs,target = self.baseline(returns,y)
            return outputs,target

        elif(mode == "prediction"):

            # Compute before last state
            irrelevant_output, cell = self.get_distrib(
                returns[:-1].to(self.device))

            batch_size = returns.shape[1]
            input_ = returns[-1:].to(self.device)

            return self.prediction_wrapper(batch_size,
                                        predictions,
                                        binary_predictions,
                                        n_bins,
                                        binning,
                                        (input_,cell),
                                        n_steps)


        elif(mode == "1 step"):

            outputs, cell = self.get_distrib(returns)
            # TODO: check this line
            outputs = outputs.squeeze(0)
            return outputs,None

        else:
            raise Exception("No such mode")

    # Forward method for iterated prediction with probabilistic inference
    # Input has shape (model_input_length,batch_size,n_dim).
    # Output has shape (nSteps,batch_size,?)
    def forward_prediction_sample(self, returns, cell,n_steps,mode="normal"):

        batch_size = returns.shape[1]
        sample_returns = returns

        # Last input is used for converting returns to values
        return_product = torch.ones(1,batch_size, 1).to(self.device)
        outputs = torch.zeros(batch_size,0,1).to(self.device)
        for j in range(n_steps):

            next_distrib, cell = self.get_distrib(sample_returns, cell)
            next_distrib = next_distrib.squeeze(2)

            # Sample from next distrib
            if mode == "normal":
                sample_returns = sample_from_distrib(next_distrib,batch_size,j)
            else:
                sample_returns = sample_from_distrib_reparametrized(next_distrib,
                                                                         batch_size,
                                                                         j)

            rescaled_sample_return = (sample_returns * self.dev + self.mean)
            return_product = return_product * rescaled_sample_return

            # Cat distrib
            outputs = torch.cat([outputs, return_product.permute(1,0,2)], 1)

        target = pyro.sample("target", pyro.distributions.Delta(return_product))

        return outputs,target

    # Forward with teacher forcing or scheduled sampling
    # Input has shape (input_length,batch_size,input_dim)
    # y has shape (batch_size,output_length,dim)
    # Output has shape (3,batch_size,output_length,self.K)
    def baseline(self,input0,y):

        #  Transpose y for teacher forcing
        # Shape is now (length,batch_size,dim)
        transposed_y = y.permute(1,0,2)

        # Preprocess input - returns has shape (input_length-1,batch_size,input_dim)
        returns = self.preprocessing(input0)
        # Outputs has shape (3,batch_size,1,self.K)
        outputs, cell = self.get_distrib(returns)

        for j in range(1,self.n_steps):

            # Do one step prediction with ground truth token
            # next_distrib has shape (3,batch_size,1,n_components)
            next_distrib, cell = self.get_distrib(transposed_y[j-1:j], cell)

            # Cat distrib
            outputs = torch.cat([outputs,next_distrib],2)

        return outputs,None

    # Forward method that returns a distribution
    # Can be stateful or not
    def get_distrib(self,returns,cell=None):

        # Feed to LSTM
        if cell is not None:
            lstm_out, (h_n,c_n) = self.lstm(returns,cell)
        else:
            lstm_out, (h_n,c_n) = self.lstm(returns)

        locs = torch.transpose(self.output_layer_loc(h_n),0,1)
        scales = torch.transpose(self.output_layer_scale(h_n),0,1)
        scales = torch.exp(scales)
        logits = torch.transpose(self.output_layer_logit(h_n),0,1)
        logits = nn.functional.softmax(logits,dim=2)

        stacked = torch.stack([locs,scales,logits])

        return stacked,(h_n,c_n)

    def forward_conditioned(self,returns,cell,args,n_steps):

        batch_size = returns.shape[1]
        sample_returns = returns

        # Last input is used for converting returns to values
        return_product = torch.ones(1, batch_size, 1).to(self.device)
        outputs = torch.zeros(batch_size, 0, 1).to(self.device)

        for j in range(n_steps):

            next_distrib, cell = self.get_distrib(sample_returns, cell)
            next_distrib = next_distrib.squeeze(2)

            if j == args.step_condition-1:

                # Get likelihood of value
                loss = Loss()

                value = args.value_condition / outputs[:,j - 1:j]
                rescaled_value = (value.squeeze(1) - self.mean) / self.dev

                weights = loss.compute_probs(next_distrib,rescaled_value).squeeze(1)

                # Next sample returns is the value
                sample_returns = args.value_condition

            else:

                # Sample from next distrib
                sample_returns = sample_from_distrib_reparametrized(next_distrib,
                                                                    batch_size,
                                                                    j)

                rescaled_sample_return = (sample_returns * self.dev + self.mean)
                return_product = return_product * rescaled_sample_return

                # Cat distrib
                outputs = torch.cat([outputs, return_product.permute(1, 0, 2)], 1)

        return outputs, weights

    def forward_log_prob(self, returns, cell, args, n_steps,outputs,conditioned=True):

        batch_size = returns.shape[1]
        sample_returns = returns

        probabilities = torch.ones(batch_size).to(self.device)
        for j in range(n_steps):

            next_distrib, cell = self.get_distrib(sample_returns, cell)
            next_distrib = next_distrib.squeeze(2)

            if conditioned and j == args.step_condition - 1:
                value = args.value_condition/outputs[:,j-1:j]
                normalized_value = (value - self.mean) / self.dev
                rescaled_value = normalized_value.squeeze(1)
            else:
                if j == 0:
                    value = outputs[:,j:j+1]
                else:
                    value = outputs[:,j:j+1]/outputs[:,j-1:j]
                normalized_value = (value - self.mean)/self.dev
                rescaled_value = normalized_value.squeeze(1)
                sample_returns = normalized_value.transpose(1, 0)

            loss = Loss()
            aux = loss.compute_probs(next_distrib, rescaled_value).squeeze(1)
            probabilities *= aux

        return torch.log(probabilities)

    def forward_reparametrized(self,input,steps,batch_size):

        returns = self.preprocessing(input)

        irrelevant_output, cell = self.get_distrib(
            returns[:-1].to(self.device))

        mean = torch.zeros(input.shape[1],device=self.device)
        for i in range(batch_size):
            outputs,_ = self.forward_prediction_sample(returns[-1:].to(self.device),
                                             cell,
                                             steps,
                                             mode="reparam")

            mean += outputs[:,-1].squeeze(1)

        mean /= float(batch_size)

        return mean