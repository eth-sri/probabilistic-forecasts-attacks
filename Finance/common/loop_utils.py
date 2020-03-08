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

import sys
sys.path.insert(1, 'models')

from model_lstm import LSTMModel
from model_mdn import MDNModel
from model_tcn import TCNModel


def get_tcn_subindex(config):
    subindex = 1000 * config["model"]["kernel size"] + \
    10 * config["model"]["channels"] + \
    config["model"]["n layers"]

    return subindex


def generate_model_index(config):

    if (config["model"]["type"] == "density tcn"):
        parallel_reps = 4
        model_index = config["model"]["n components"]

        model_index = 10000 * config["model"]["n components"] + get_tcn_subindex(config)


    elif (config["model"]["type"] == "tcn"):

        if config["model"]["channels"] > 20:
            parallel_reps = 2
        elif config["model"]["channels"] > 10:
            parallel_reps = 2
        else:
            parallel_reps = 4

        model_index = get_tcn_subindex(config)

    elif (config["model"]["type"] == "mdn"):
        parallel_reps = 4
        model_index = config["model"]["n components"]

    else:
        parallel_reps = 4
        model_index = 0

    return parallel_reps,model_index


def generate_model(datareader, conf, output_length):

        if conf["type"]=="mdn":
                    model = MDNModel(datareader.meta["mean"],
                                          datareader.meta["std"],
                                          conf,
                                          output_length)
        elif conf["type"] == "tcn":
            model = TCNModel(datareader.meta["mean"],
                                  datareader.meta["std"],
                                  output_length,
                                  conf["n layers"],
                                  conf["channels"],
                                  conf["kernel size"])

        else:
            model = LSTMModel(datareader.meta["mean"],
                                  datareader.meta["std"],
                                 output_length)

        return model