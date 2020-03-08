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

from model import Model
from TCN_locus_labs import TemporalBlock


class TCNBasedModel(Model):

    def __init__(self,mean_return,dev_return,n_steps,n_layers,n_channels,kernel_size):
        super(TCNBasedModel, self).__init__(mean_return,dev_return)

        self.n_steps = n_steps
        self.n_layers = n_layers
        self.n_channels = n_channels

        layers = []

        for i in range(n_layers):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else n_channels
            out_channels = n_channels
            layers += [TemporalBlock(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=1,
                                    dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                    dropout=0.)]


        self.network = nn.Sequential(*layers)
