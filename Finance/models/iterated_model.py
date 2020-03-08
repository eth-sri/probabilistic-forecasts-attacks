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
import torch

from common import utils


class IteratedModel(nn.Module):

    # oneStepModel is instance of model.Model, and already trained
    def __init__(self, oneStepModel, nSteps, params,cpu=False):
        super(IteratedModel, self).__init__()

        if cpu:
            self.device = torch.device("cpu")
        else:
            self.device = utils.choose_device()

        self.oneStepModel = oneStepModel
        self.mean = self.oneStepModel.mean
        self.std = self.oneStepModel.dev
        self.nSteps = nSteps
        self.params = params

        # Disable parameters
        self.disable_parameters()

    def disable_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    # Function to print parameters of a model
    def print_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                print(param.data)