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

import numpy as np
from scipy.stats import norm

# A prediction should inherit this class
# It should have a __call__ method that takes as input a sample prediction of shape
# (batch_size,output_length)
class Prediction:

    def __init__(self):
        print("_")


class CumulatedReturn(Prediction):

    def __init__(self,step,n_bins):

        self.step = step
        self.price = 0
        self.name = "Cumulated Return"
        self.bins = np.linspace(0.95,1.05,n_bins-1)

    def __call__(self,prediction):

        return prediction[:,self.step-1]


class EuropeanPutOptionValue(Prediction):

    def __init__(self,step,price,n_bins):

        self.step = step
        self.price = price
        self.name = "European Put Option Value"
        self.bins = np.linspace(0, 0.2, n_bins-1)

    def __call__(self,prediction):

        return np.maximum(0,prediction[:,self.step-1]-self.price)


class EuropeanCallOptionValue(Prediction):

    def __init__(self,step,price,n_bins):

        self.step = step
        self.price = price
        self.name = "European Call Option Value"
        self.bins = np.linspace(0, 0.5, n_bins-1)

    def __call__(self,prediction):

        return np.maximum(0,self.price-prediction[:,self.step-1])


# Compute the probability that a Limit Order for Buying will be executed in
# the next time steps. If the price is more than 1, then you expect i to be
# always one
class LimitBuyOrder(Prediction):

    def __init__(self, step, price,n_bins):
        self.step = step
        self.price = price
        self.name = "Limit Buy Order - Proba of Success"
        self.bins = np.array([0.5])

    def __call__(self, prediction):
        return np.any(prediction[:,:self.step]<=self.price, axis=1)


# Same for Limit Sell Order, if price is less than 1, always 1
class LimitSellOrder(Prediction):

    def __init__(self, step, price,n_bins):
        self.step = step
        self.price = price
        self.name = "Limit Sell Order - Proba of Success"
        self.bins = np.array([0.5])

    def __call__(self, prediction):
        return np.any(prediction[:,:self.step]>=self.price, axis=1)


def construct_predictions_list(steps,n_bins):

    predictions = []
    binary_predictions = []

    # These should always come first !!!!
    predictions += [CumulatedReturn(step,n_bins) for step in steps]

    max_step = steps[-1]
    thresholds = [0.9, 1, 1.1]
    for threshold in thresholds:
        predictions.append(EuropeanCallOptionValue(max_step,threshold,n_bins))
        predictions.append(EuropeanPutOptionValue(max_step, threshold,n_bins))

    sell_thresholds = [1.01,1.05,1.2]
    for threshold in sell_thresholds:
        p = LimitSellOrder(max_step,threshold,n_bins)
        binary_predictions.append(p)

    buy_thresholds = [0.8,0.95,0.99]
    for threshold in buy_thresholds:
        p = LimitBuyOrder(max_step, threshold, n_bins)
        binary_predictions.append(p)

    # Append trading strategies at the end
    predictions_dict = {
        "eval": predictions,
        "names": [pred.name for pred in predictions],
        "steps": [pred.step for pred in predictions],
        "prices": [pred.price for pred in predictions]
    }

    binary_predictions_dict = {
        "eval":binary_predictions,
        "names": [pred.name for pred in binary_predictions],
        "steps": [pred.step for pred in binary_predictions],
        "prices": [pred.price for pred in binary_predictions]
    }

    return predictions_dict,binary_predictions_dict

def get_cdf_thresholds():
    radius = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    cdf = [norm.cdf(r) for r in radius]
    return radius,cdf



