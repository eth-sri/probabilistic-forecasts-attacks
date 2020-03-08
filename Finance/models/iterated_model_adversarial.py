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

from common import utils
from models.iterated_model import IteratedModel

# module for adversarial attack
class AdversarialAttackModel(IteratedModel):

    # oneStepModel is instance of model.Model, and already trained
    def __init__(self,batch_size,oneStepModel,nSteps,params):
        super(AdversarialAttackModel, self).__init__(oneStepModel,nSteps,params)

        # Perturbation, to initialize
        # Has size (nSteps,)
        self.perturbation = nn.Parameter(torch.zeros(self.nSteps, 1,batch_size, 1))

        # Print remaining parameters
        #self.print_parameters()

        self.modify = utils.modify_return(params)

    # Input has shape (model_input_length,batch_size,input_dim).
    # Represents a vector of perturbation of the outputs
    def forward(self,input):

        sliding_window = input

        # Last input is used for converting returns to values
        value = sliding_window[-1:]

        for i in range(self.nSteps):

            # Do one step prediction
            # next_return has shape (batch_size,output_length,output_dim)
            next_return = self.oneStepModel(sliding_window)
            modified_return = self.modify.apply(next_return)
            rescaled_return = modified_return * self.std + self.mean
            rescaled_return = rescaled_return.permute(1,0,2)

            # Transformed to value
            value = value * rescaled_return

            # Add next_value and perturbation
            value = value+self.perturbation[i]

            # Append sliding_window and next_value
            sliding_window = torch.cat([sliding_window[1:],value],dim=0)

        result =  sliding_window[-self.nSteps:]

        # Return last nSteps
        return result