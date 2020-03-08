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

from tcn_based_model import TCNBasedModel


class TCNModel(TCNBasedModel):

    def __init__(self,mean_return,dev_return,n_steps,n_layers,n_channels,kernel_size):
        super(TCNModel, self).__init__(mean_return,dev_return,n_steps,
                                       n_layers,n_channels,kernel_size)

        self.linear = nn.Linear(n_channels, 1)

    # Returns has shape (batch_size,n_channels,input_length)
    # The last dimension is left_padded to min_input_length
    def pad_sequence_to_length(self,returns):

        padded_returns = torch.zeros(returns.shape[0],
                                     returns.shape[1],
                                     self.min_input_length).to(self.device)


        padded_returns[:,:,-returns.shape[2]:] = returns

        return padded_returns

    def forward_aux(self,padded_returns):

        aux = self.network(padded_returns)

        return self.linear(aux.transpose(1,2))

    # Input has shape (window_length,batch_size,input_dim)
    # Should be reshaped, because TCN is not batch first
    def forward(self,input,mode,
                n_bins=None,
                y=None,
                n_steps=None,
                predictions=None,
                binary_predictions=None,
                binning=False):
        # Preprocess input

        returns = self.preprocessing(input)

        reshaped_returns = returns.permute(1,2,0)

        # Pad second dimension to at least min_input_length
        # padded_returns = self.pad_sequence_to_length(reshaped_returns)

        if mode == "teacher forcing":

            # y has shape (batch_size,output_length,1)
            transposed_y = y.permute(0,2,1)

            # Append padded_returns and y
            input = torch.cat([reshaped_returns,transposed_y],dim=2)
            output = self.forward_aux(input)[:,-self.n_steps-1:-1]

            return output, None

        elif mode == "prediction":

            # output will have shape (batch_size,seq_len,1)
            output = torch.ones(input.shape[1],0,1,device=self.device)
            network_input = reshaped_returns
            return_product = torch.ones(input.shape[1],1,1,device=self.device)

            for i in range(n_steps):

                # output has shape (batch_size,1,1)
                out = self.forward_aux(network_input)[:,-1:]

                # Cat output and input
                network_input = torch.cat([network_input,out],dim=2)

                # Gather output in array
                rescaled_return = self.rescale_return(out)
                return_product *= rescaled_return

                output = torch.cat([output, return_product], dim=1)

            length = len(self.log_samples)
            if binning:

                a = self.compute_predictions(output, predictions, n_bins)
                b = self.compute_predictions(output, binary_predictions, 2)

                return [a] * length, [b] * length

            return [output]*length

        elif mode == "1 step":

            input = reshaped_returns
            output = self.forward_aux(input)[:,-1]


            return output,None


