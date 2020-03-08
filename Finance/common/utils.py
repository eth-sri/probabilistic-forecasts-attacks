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
import json
import numpy as np
import math
import os


ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

# Function to transform tensor into returns
def compute_returns_tensor(input,window_length):

    # Shift Tensor, and divide to obtain returns
    # (See https://discuss.pytorch.org/t/implementation-of-function-like-numpy-roll/964/5)
    narrowed_without_last = torch.narrow(input,0,0,window_length-1)
    narrowed_without_first = torch.narrow(input,0,1,window_length-1)
    returns = narrowed_without_first/narrowed_without_last
    return returns


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self,input,y):
        means = input[0]
        logits = input[2]
        expectation = means * logits
        output = torch.sum(expectation, dim=2).unsqueeze(2)
        return self.MSE(output,y)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def compute_probs(self,input,y):

        locs = input[0]
        scales = input[1]

        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((y - locs) / scales) ** 2) / scales

        # Add epsilon > 0 to avoid underflows
        ret += 1e-8

        return ret

    def compute_probabilities_per_element(self,input_,y):

        # Compute probabilities
        probs = self.compute_probs(input_, y)

        # Scale probabilities with logits
        logits = input_[2]
        probs = logits * probs

        # Sum weighted probabilities
        aux = torch.sum(probs, dim=2)

        return aux

    # Input has shape (3,batch_size,output_length,n_components)
    # y has shape (batch_size,output_length)
    # Output has shape (batch_size)
    def forward(self,input_,y):

        aux = self.compute_probs(input_,y)

        nll = -torch.log(aux)

        return torch.mean(nll)

def choose_loss(model):

    if model["type"] == "mdn":
        loss = Loss()
    elif model["type"] == "lstm":
        loss_name = model["loss"]
        if loss_name == "MSE":
            loss = nn.MSELoss()
        elif loss_name == "MAE":
            loss = nn.L1Loss()
        else:
            raise Exception("No such loss")
    elif model["type"] == "tcn":
        loss = nn.MSELoss()
    elif model["type"] == "density tcn":
        loss = Loss()
    else:
        raise Exception("No such model")
    return loss


def choose_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Reads args for training
def read_exp_file(args):
    a = read_json(args.file)
    return a["exp_params"],a["training_params"]


def read_json(file):
    with open(file) as f:
        a = json.load(f)
    return a


# Functions that can be returned by  modify return
class Square():

    def __init__(self,factor,params):
        self.factor = factor

    def apply(self,return_):
        if(isinstance(return_,np.ndarray)):
            modified_return = np.sign(return_)*return_ ** 2 * self.factor
        else:
            modified_return = torch.sign(return_)*return_ ** 2 * self.factor
        return modified_return

class Root():

    def __init__(self,factor,params):
        self.factor = factor

    def apply(self,return_):
        if(isinstance(return_,np.ndarray)):
            modified_return = np.sign(return_)*np.sqrt(np.abs(return_)) * self.factor
        else:
            modified_return = torch.sign(return_)*torch.sqrt(torch.abs(return_) ) * self.factor
        return modified_return


class Linear():

    def __init__(self, factor, params):
        self.factor = factor
        self.threshold = 0.
        if "threshold" in params["scaling"]:
            self.threshold = params["scaling"]["threshold"]

    def apply(self, return_):
        modified_return = return_ * self.factor
        if isinstance(return_, np.ndarray):
            modified_return[\
                np.logical_and(modified_return < self.threshold,\
                               modified_return > -self.threshold)] = 0.

        else:
            modified_return[(modified_return < self.threshold)
                            & (modified_return > -self.threshold)] = 0.
        return modified_return


# Returns a function that can be applied to a return
def modify_return(params):
    mode = params["scaling"]["mode"]
    factor = params["scaling"]["factor"]
    if mode == "square":
        func = Square(factor,params)
    elif mode == "linear":
        func = Linear(factor,params)
    elif mode == "root":
        func = Root(factor,params)
    else:
        raise Exception("Unsupported scaling mode")
    return func


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def convert_from_tensor(var):
    if isinstance(var, torch.Tensor):
        var = var.cpu().numpy()
    return var


def get_returns(output_day,y_day,k):

    largest_indexes = np.argpartition(output_day, -k, axis=0)[-k:]
    smallest_indexes = np.argpartition(output_day, k - 1, axis=0)[:k]

    effective_pos_returns = y_day[largest_indexes]
    effective_neg_returns = y_day[smallest_indexes]

    mean = (np.mean(effective_pos_returns) -
                            np.mean(effective_neg_returns)) / 2.

    return mean


def detect_nan(tensor):

    return torch.isnan(tensor).any()