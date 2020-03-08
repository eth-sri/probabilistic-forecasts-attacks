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

import torch.nn as nn
import numpy as np
import torch
import statsmodels.stats.proportion as st
import time

import sys
sys.path.insert(1, 'dataset')
sys.path.insert(1, 'common')
from generation import data_reader
import utils



class Model(nn.Module):
    def __init__(self,mean_return,dev_return):
        super(Model, self).__init__()

        self.mean = mean_return
        self.dev = dev_return
        self.device = utils.choose_device()


    def rescale_return(self,ret):
        rescaled_return = ret * self.dev + self.mean
        return rescaled_return

    def preprocessing(self,input):

        shape = input.shape
        window_length = shape[0]

        returns = utils.compute_returns_tensor(input, window_length)
        returns = (returns - self.mean) / self.dev

        return returns

    def compute_predictions(self,output,predictions,n_bins):

        output = output.cpu().numpy().squeeze(2)
        batch_size = output.shape[0]
        n_predictions = len(predictions["eval"])
        pred_bins = np.zeros((n_bins-1,n_predictions))

        # Prediction
        res = np.zeros((n_bins+1,batch_size,n_predictions))
        for i in range(n_predictions):
            pred_bins[:,i] = predictions["eval"][i].bins
            res[0,:,i] = predictions["eval"][i](output)

        # Efficient version
        treated = np.zeros((batch_size,n_predictions))
        for index in range(n_bins-1):
            good_index = np.logical_and(res[0,:,:] < pred_bins[index,:],
                                        np.logical_not(treated))
            res[index+1,good_index] = 1
            treated = np.logical_or(treated,good_index)

        res[n_bins,np.logical_not(treated)] = 1

        return res.transpose(1,2,0)

    # mean has shape (batch,n_predictions)
    def compute_confidence(self,mean):

        batch_size = mean.shape[0]
        n_predictions = mean.shape[1]

        confidence = np.zeros((batch_size,n_predictions))
        for i in range(batch_size):
            for j in range(n_predictions):
                if(mean[i,j] > self.n_samples/2):
                    val = mean[i,j]
                else:
                    val = self.n_samples-mean[i, j]
                confidence[i,j] = st.proportion_confint(self.n_samples-mean[i, j],
                                                  self.n_samples,
                                                  alpha=0.1,
                                                  method="beta")[0]

        certification_stats = np.zeros((len(self.cdf),batch_size,n_predictions))

        for index in range(len(self.cdf)):
            certified = confidence >= self.cdf[index]
            certification_stats[index,certified] = 1

        return certification_stats.transpose(1,2,0)

    def expand_to_cdf(self,b):
        add_shape = (b.shape[0], b.shape[1], len(self.cdf))
        c = np.zeros(add_shape)
        return np.concatenate((b, c), axis=2)

    def init_attack_mode(self):

        self.disable_parameters()

        self.print_parameters()

    def disable_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    # Function to print parameters of a model
    def print_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                print(param.data)

    def prediction_wrapper(self,batch_size,predictions,binary_predictions,n_bins,binning,
                           input_,n_steps):

        if binning:

            mean_list = [0]*len(self.log_samples)
            binary_mean_list = [0] * len(self.log_samples)

            mean = np.zeros((batch_size,
                             len(predictions["eval"]),
                             n_bins + 1))
            binary_mean = np.zeros((batch_size,
                                    len(binary_predictions["eval"]),
                                    3))

        else:
            mean_list = [0] * len(self.log_samples)
            mean = torch.zeros(batch_size,n_steps,1).to(self.device)

        if binning:
            mean_time_sample = 0.
            mean_time_binning = 0.
        for i in range(self.n_samples):

            start_time = time.time()

            sample,target = self.forward_prediction_sample(*input_,n_steps)

            if binning:

                mid_time = time.time()
                mean_time_sample += mid_time - start_time

                res = self.compute_predictions(sample, predictions, n_bins)
                mean += res

                binary_res = self.compute_predictions(sample, binary_predictions, 2)
                binary_mean += binary_res

                if i+1 in self.log_samples:
                    index = self.log_samples.index(i+1)
                    mean_list[index] = np.copy(mean)
                    binary_mean_list[index] = np.copy(binary_mean)
                    mean_list[index] /= (i+1)
                    binary_mean_list[index] /= (i+1)

                end_time = time.time()
                mean_time_binning += end_time - mid_time

            else:

                mean += sample

                if i+1 in self.log_samples:

                    index = self.log_samples.index(i+1)
                    mean_list[index] = np.copy(utils.convert_from_tensor(mean))
                    mean_list[index] /= (i+1)

        if binning:

            return mean_list, binary_mean_list


        else:
            return mean_list



class GroundTruthModel(Model):

    def __init__(self,folder):
        year = int(folder.split("_")[-1])

        self.datareader = data_reader.DataReader(year, torch.device('cpu'))

    def forward(self, input0,mode,n_bins=None,y=None,
                n_steps=None,predictions=None,
                binary_predictions=None,binning=False):

        length = len(self.log_samples)
        if binning:
            a = self.compute_predictions(y, predictions, n_bins)
            b = self.compute_predictions(y, binary_predictions, 2)

            return [a] * length, [b] * length

        return [y] * length

class ClimatologicalModel(Model):

    def __init__(self,folder,rgt,rgt_binary):
        year = int(folder.split("_")[-1])

        self.datareader = data_reader.DataReader(year, torch.device('cpu'))
        self.rgt = rgt
        self.rgt_binary = rgt_binary

    def forward(self, input0,mode,n_bins=None,y=None,
                n_steps=None,predictions=None,binary_predictions=None,
                binning=False):

        length = len(self.log_samples)

        batch_size = input0.shape[1]

        if binning:
            a = np.broadcast_to(self.rgt[:, :-1],
                                (batch_size, len(predictions["eval"]), n_bins + 1))

            b = np.broadcast_to(self.rgt_binary[:,:3],
                                (batch_size,len(binary_predictions["eval"]),3))




            return [a] * length, [b] * length

        a = np.broadcast_to(self.rgt[:, :-1],
                                (batch_size, self.rgt.shape[2]))
        return [a] * length





