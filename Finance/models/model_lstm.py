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

from lstm_based_model import LstmBasedModel

class LSTMModel(LstmBasedModel):
    def __init__(self,mean_return,dev_return,n_steps):
        super(LSTMModel, self).__init__(mean_return,dev_return)

        self.n_steps = n_steps

        # LSTM output layer (investigate with activation functions ?)
        self.output_layer = nn.Linear(self.hidden_dim,1)

    def forward(self, input0,mode,
                n_bins=None,
                attack=False,
                y=None,
                n_steps=None,
                predictions=None,
                binary_predictions=None,
                binning=False):

        # Preprocess input
        returns = self.preprocessing(input0)
        next_return, cell = self.get_output(returns)

        if mode == "teacher forcing":
            output = next_return
            transposed_y = y.permute(1, 0, 2)
            n_steps = self.n_steps

        elif mode == "prediction":
            rescaled_return = self.rescale_return(next_return)
            return_product = rescaled_return
            output = return_product


        elif(mode == "1 step"):
            output = next_return
            return output,None

        for j in range(1,n_steps):

            # Do one step prediction
            # next_return has shape
            if mode == "teacher forcing":
                next_return, cell = self.get_output(transposed_y[j-1:j], cell)
                output = torch.cat([output, next_return], dim=1)

            elif mode == "prediction":
                next_return, cell = self.get_output(next_return.permute(1,0,2), cell)
                rescaled_return = self.rescale_return(next_return)
                return_product *= rescaled_return
                output = torch.cat([output, return_product], dim=1)


        if mode == "prediction":
            if attack:
                return output

            length = len(self.log_samples)

            if binning:
                a = self.compute_predictions(output, predictions, n_bins)
                b = self.compute_predictions(output, binary_predictions, 2)


                return [a] * length, [b] * length

            return [output]*length

        return output,None

    # Forward method that returns a distribution
    # Can be stateful or not
    # Output has dim (batch_size,seq_len,dim)
    def get_output(self, returns, cell=None):

        # Feed to LSTM
        if cell is not None:
            lstm_out, (h_n, c_n) = self.lstm(returns, cell)
        else:
            lstm_out, (h_n, c_n) = self.lstm(returns)
        outputs = torch.transpose(self.output_layer(h_n), 0, 1)

        return outputs, (h_n, c_n)
