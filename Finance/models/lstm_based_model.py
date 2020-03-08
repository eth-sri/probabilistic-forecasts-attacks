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

import model
import torch.nn as nn

class LstmBasedModel(model.Model):
    def __init__(self,mean_return,dev_return):
        super(LstmBasedModel, self).__init__(mean_return,dev_return)

        # LSTM layer
        self.input_dim = 1
        self.hidden_dim = 25
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)