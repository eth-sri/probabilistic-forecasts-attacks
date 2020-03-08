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

import argparse
import numpy as np
import json
import os
import itertools
import pandas as pd
import time
from joblib import Parallel, delayed

import sys
sys.path.insert(1, 'common')
from project_structure import *
import utils, loop_utils
from training import Training


class TrainingLoop:

    def __init__(self,exp_params,training_params,debug):

        self.exp_params = exp_params
        self.training_params = training_params
        self.n_reps = self.exp_params["reps"]
        self.debug = debug

        # Create folder in training logs, with exp name
        folder_name = exp_params["name"]+"/"
        self.folder_path = training_folder+folder_name

        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)

    def log_meta(self,time,folder):
        # Log meta file with experiments parameters
        stats = {"Time per run (seconds)":time}
        with open(folder + "meta.json", "w") as f:
            f.write(json.dumps(self.exp_params, indent=4))
            f.write("\n")
            f.write(json.dumps(self.training_params, indent=4))
            f.write("\n")
            f.write(json.dumps(stats,indent=4))

    def unfold(self, entry):

        result = []

        keys = [k for k in list(entry.keys())]
        vals = [entry[key] for key in keys]
        for i in range(len(vals)):
            if (not isinstance(vals[i], list)):
                vals[i] = [vals[i]]
        product = list(itertools.product(*vals))

        # Add every config to product
        for config_aux in product:

            config = {}

            for (k, v) in zip(keys, config_aux):
                config[k] = v

            result.append(config)

        return result

    def log_config(self,config,folder,time):

        json_file = folder + "parameters.json"
        config["Mean time per repetition (seconds)"] = time
        with open(json_file, 'w') as outfile:
            json.dump(config, outfile, indent=4)
        return

    def run(self):

        # For each possible combination of parameters
        list_configs = self.unfold(self.training_params)
        print("List configs")
        print(list_configs)
        print("\n")

        # Init result dataframe
        length = len(list_configs)
        models = np.zeros((length,3))
        metrics = np.zeros((length,self.n_reps,7))

        start_time = time.time()

        for year in exp_params["Years"]:

            print("\n")
            print("Year",year)

            year_folder = self.folder_path + "Year_"+str(year)+"/"
            if not os.path.exists(year_folder):
                os.mkdir(year_folder)
            for config_index in range(length):

                print("Config index",config_index+1,"/",length)

                config = list_configs[config_index]

                parallel_reps,model_index = loop_utils.generate_model_index(config)

                models[config_index] = [
                    model_index,
                    config["lr"],
                    config["batch"]
                ]

                # Create a subfolder
                subfolder = year_folder+"config_"+str(config_index)+"/"
                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)

                start_time_inner = time.time()

                def repetition(i):
                    folder = subfolder + "run_" + str(i) + "/"
                    t = Training(
                        year,
                        config,
                        folder,
                        self.debug)
                    return t.train()

                results = Parallel(n_jobs=parallel_reps)(delayed(repetition)(i) for i in range(self.n_reps))

                for i in range(self.n_reps):
                    metrics[config_index,i] = results[i]

                end_time_inner = time.time()
                elapsed_time_inner = round((end_time_inner - start_time_inner) /
                                           float(self.n_reps), 2)

                self.log_config(config, subfolder, elapsed_time_inner)

            # Compute mean time per config and rep, and round
            end_time = time.time()
            elapsed_time = round((end_time-start_time)/float(self.n_reps*length),2)

            # Get stats for these 10 runs
            # Average/Std final val loss
            means = np.mean(metrics,axis=1)
            stds = np.std(metrics,axis=1)


            data = np.concatenate((models,means,stds),axis=1)
            # Format results, and log it to a file
            columns = [
                "Model",
                "LR",
                "Batch",
                "M Epochs",
                "M training loss (MSE)",
                "M test loss (MSE)",
                "M std",
                "M daily result",
                "M training loss (MDN)",
                "M test loss (MDN)",
                "S Epochs",
                "S training loss (MSE)",
                "S test loss (MSE)",
                "S std",
                "S daily result",
                "S training loss (MDN)",
                "S test loss (MDN)"
            ]
            df = pd.DataFrame(columns=columns,
                              data=data)
            df.to_csv(year_folder+"result.csv",index=True)

            self.log_meta(elapsed_time,year_folder)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='File with experiment parameters')
    parser.add_argument('--debug', action="store_true", help='Debug mode')
    args = parser.parse_args()

    exp_params,training_params = utils.read_exp_file(args)


    loop = TrainingLoop(exp_params,training_params,args.debug)
    loop.run()

