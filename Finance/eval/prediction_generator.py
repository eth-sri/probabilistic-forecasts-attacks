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
import numpy as np

import sys
sys.path.insert(1, 'common')
import utils


class Prediction_Generator:

    def __init__(self,model,steps,k,batch_days,samples,
                 rps=False,
                 predictions=None,
                 binary_predictions = None,
                 n_bins = 0,
                 debug=False):

        self.model = model
        self.steps = steps
        self.k = k
        self.batch_days = batch_days
        self.samples = samples

        self.rps = False
        if rps:
            self.rps = True
            self.predictions = predictions
            self.binary_predictions = binary_predictions

            self.n_bins = n_bins

        self.debug = debug

    def append_rps(self,output,y,n_bins,pred):

        shape = output.shape
        shape = (shape[0],shape[1],shape[2]+1)
        shape_y = (shape[0], shape[1], n_bins)
        res = np.zeros(shape)
        res[:,:,:output.shape[2]] = output

        # Do cumsum on output
        cumsum_output = np.cumsum(output[:,:,1:n_bins+1],axis=2)

        # Do cumsum on y
        y_index = np.zeros(shape_y)

        for i in range(len(pred["eval"])):

            aux = pred["eval"][i](y.cpu().numpy())

            for j in range(shape[0]):
                index = np.searchsorted(pred["eval"][i].bins,aux[j], side='right')
                y_index[j,i,index] += 1

        cumsum_y = np.cumsum(y_index,axis=2)

        squared_diff = (cumsum_y - cumsum_output)**2

        sum_squared_diff = np.sum(squared_diff,axis=2)

        res[:,:,n_bins+1] = sum_squared_diff
        return res

    def compute_predictions(self):

        with torch.no_grad():

            n_days = self.model.datareader.n_test_batches

            start_days = 0
            if self.debug:
                start_days = 0
                n_days = 10
                self.batch_days = 10

            if self.rps :
                rps_results = np.zeros((len(self.samples),
                                        len(self.predictions["eval"]),
                                        self.n_bins+2))
                binary_results = np.zeros((len(self.samples),
                                           len(self.binary_predictions["eval"]),
                                           4))

            trading_results = np.zeros((len(self.samples),
                                       len(self.k),
                                       len(self.steps)))

            for i in range(start_days, n_days, self.batch_days):

                bound = min(self.batch_days, n_days - i)
                # Get output (feed batches 10 by 10)
                x_cat = []
                y_cat = []
                for j in range(bound):

                    x, y = self.model.datareader.get_test_batch(i + j ,cumulative=True)
                    x_cat.append(x)
                    y_cat.append(y)

                n_per_day = x_cat[0].shape[1]

                x_cat = torch.cat(x_cat, dim=1)
                y_cat = torch.cat(y_cat, dim=0)

                if self.rps:

                    oc,boc = self.model.forward(x_cat,mode="prediction",
                                                       y=y_cat,
                                                       n_bins=self.n_bins,
                                                       n_steps=max(self.steps),
                                                       predictions=self.predictions,
                                                       binary_predictions=self.binary_predictions,
                                                       binning=True)

                else:
                    oc = self.model.forward(x_cat, mode="prediction",
                                                y=y_cat,
                                                n_steps=max(self.steps))

                # For each number of samples
                for m in range(len(self.samples)):

                    if self.rps:
                        output_cat = self.append_rps(oc[m],
                                                     y_cat,
                                                     self.n_bins,
                                                     self.predictions)
                        binary_output_cat = self.append_rps(boc[m],
                                                            y_cat,
                                                            2,
                                                            self.binary_predictions)
                    else:
                        output_cat = oc[m]

                    for j in range(bound):
                        left = n_per_day * j
                        right = n_per_day * (j + 1)

                        output_day_j = output_cat[left:right]

                        if self.rps:
                            rps_results[m] += np.mean(output_day_j,axis=0)
                            binary_results[m] += np.mean(binary_output_cat[left:right],axis=0)

                        else:
                            # Compute trading results
                            for l in range(len(self.steps)):

                                steps = self.steps[l]
                                y = y_cat[left:right, steps - 1].view(-1).cpu().numpy()

                                # Cumulated return eval should always be put first
                                # Adding the 0 for second dimension
                                returns = output_day_j[:, self.steps[l] - 1, 0]
                                returns = utils.convert_from_tensor(returns)

                                for n in range(len(self.k)):
                                    aux = utils.get_returns(returns, y, self.k[n])
                                    trading_results[m, n, l] += aux

            denominator = float(n_days-start_days)
            trading_results /= denominator
            if self.rps:
                rps_results /= denominator
                binary_results /= denominator

                return rps_results,binary_results,trading_results

            return trading_results


