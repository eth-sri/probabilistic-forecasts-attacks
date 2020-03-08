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
import os
import torch
import time
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'common')
sys.path.insert(0, 'dataset/generation')
sys.path.insert(0, 'models')
import data_reader
from iterated_prediction_loop import IteratedPredictionLoop
from project_structure import *
import utils
import attack_generator as atg


class AttackLoop(IteratedPredictionLoop):

    def __init__(self,args):

        super(AttackLoop,self).__init__(args,attack_log_folder)

        self.batch_size = args.batch_size
        self.n_iterations = args.n_iterations
        self.c = args.c
        self.lr = args.learning_rate
        self.n_samples = args.samples
        self.print = args.print

        if not isinstance(self.batch_size, list):
            self.batch_size = [self.batch_size]

        if not isinstance(self.c, list):
            self.c = [self.c]

        if not isinstance(args.max_pert, list):
            args.max_pert = [args.max_pert]

        if self.batched_days == 0:
            print("Adaptive number of batched days")

        if args.fixed_epsilon:
            self.indexes = ["estimator", "batch size", "lr", "n iterations"]
            if args.target == "regression":
                self.indexes += ["k"]
            self.indexes += ["max pert"]
        else:
            self.indexes = ["c", "batch size", "lr", "n iterations"]
        self.log_batch = args.log_batch
        self.args = args

        if self.args.target=="regression":
            # Check that not a conditional
            assert( not self.args.conditional)

            if not isinstance(self.args.k,list):
                self.args.k = [self.args.k]

    def get_config_folders(self, year_folder):
        config_folders = self.list_dir(os.path.join(self.folder, year_folder))

        config_list = []
        for c in config_folders:
            config, folder_full, run_folders, n_reps = self.load_config(year_folder, c)
            config_list.append({"n reps": n_reps,
                                "training folder": folder_full,
                                "output folder": "mdn",
                                "run folders": run_folders,
                                "config": config})
        # Iterate over config
        return config_list

    def create_dataframe(self,data,output_folder):

        if self.args.fixed_epsilon:
            if self.args.target == "binary":
                columns = ["estimator", "batch size", "lr", "n iterations","max pert","buy","sell"]
            else:
                columns = ["estimator", "batch size", "lr","n iterations",
                           "k","max pert","return"]

        else:
            columns = ["c", "batch size", "lr", "n iterations","mse","distance"]+\
                      ["norm step "+str(i) for i in range(241)]

        dataframe = pd.DataFrame(data=data, columns=columns)

        dataframe.set_index(self.indexes, inplace=True)
        dataframe.sort_index(inplace=True)

        dataframe.to_csv(output_folder+"/results.csv")

        return dataframe

    def create_log_dataframe(self, logs):

        columns = ["norm", "distance", "loss"]

        dataframe = pd.DataFrame(data=logs, columns=columns)

        return dataframe

    def compute_mean(self,folder):

        year = int(folder.split("_")[-1])

        datar = data_reader.DataReader(year, torch.device('cpu'))

        n_days = datar.n_test_batches

        values = []

        for i in range(n_days):
            x, y = datar.get_test_batch(i , cumulative=True)

            if self.args.conditional:
                y_step = y[:,self.args.step_prediction-1]
            else:
                y_step = y[:,self.args.steps-1]

            values.append(utils.convert_from_tensor(y_step))

        values = np.array(values)
        mean = values.mean()
        return mean

    def get_batch_days(self,batch_size):

        if batch_size <= 2:
            return 128
        if batch_size <= 20:
            return 85
        if batch_size <= 50:
            return 50
        if batch_size <= 100:
            return 25
        if batch_size <= 200:
            return 15
        else:
            return 10

        return batched_days

    # Config contains information about the model attacked
    # c coefficient is only for fixed c attack
    # batch size is only used for
    def iteration(self,config,c,batch_size,keys,log_dataframes,
                  data,i,mean_step_return,estimator):

        run_folder = os.path.join(config["training folder"],
                                  config["run folders"][i])
        model = torch.load(os.path.join(run_folder, 'model.pt'),
                           map_location=self.device)

        # Update parameters for model
        model.device = utils.choose_device()
        model.datareader.device = utils.choose_device()
        model.n_samples = self.n_samples

        if self.args.fixed_epsilon:
            attack_gen = atg.FixedEpsilonAttackGenerator(model,
                                                         args,
                                                         mean_step_return,
                                                         estimator)

            if self.args.target == "binary":
                mean_percentage = attack_gen.compute_predictions()

                for l in range(len(self.args.max_pert)):
                    values = [mean_percentage["buy"][l], mean_percentage["sell"][l]]

                    line = [estimator, batch_size, self.lr, self.n_iterations,
                            self.args.max_pert[l]] + values
                    data.append(line)
            else:
                mean_perturbed = attack_gen.compute_predictions()

                for m in range(len(self.args.k)):
                    for l in range(len(self.args.max_pert)):
                        values = [mean_perturbed[m][l]]

                        line = [estimator, batch_size, self.lr, self.n_iterations,
                                self.args.k[m],
                                self.args.max_pert[l]] + values
                        data.append(line)


        else:

            if self.args.log_batch:
                batched_days = self.get_batch_days(batch_size)
            else:
                batched_days = self.args.days

            attack_gen = atg.FixedCAttackGenerator(model,
                                                   args,
                                                   mean_step_return,
                                                   c,
                                                   batch_size,
                                                   batched_days=batched_days)

            # Second result is certification statistics
            if self.args.log_batch:
                values,logs = attack_gen.compute_predictions()

                df = self.create_log_dataframe(logs)
                key = (c, batch_size)
                keys.append(key)

                # Create dataframe with logs
                log_dataframes.append(df)

            else:
                values = attack_gen.compute_predictions()

            line = [c, batch_size, self.lr, self.n_iterations] + values
            data.append(line)


    def run(self):

        for year_folder in self.year_folders:

            # Compute mean for this year
            mean_step_return = self.compute_mean(year_folder)

            year_output_folder = self.make_output_folder(year_folder)
            config_list = self.get_config_folders(year_folder)

            for config in config_list:

                print("\n")
                print("Model", config["output folder"])

                # Only one model per year for attack
                for i in range(1):

                    config_output_full = os.path.join(year_output_folder,
                                                      config["output folder"])

                    utils.makedir(config_output_full)
                    output_folder = os.path.join(config_output_full,
                                                 config["run folders"][i])
                    utils.makedir(output_folder)

                    log_dataframes = []
                    keys = []

                    data = []

                    if self.args.fixed_epsilon:
                        c_iter = [0]
                    else:
                        c_iter = self.c

                    for c in c_iter:

                        for batch_size in self.batch_size:

                            if self.args.log_batch:
                                print("Batch size",batch_size)

                            if self.args.fixed_epsilon and not self.args.conditional:
                                est_iter = ["score","reparam"]
                            else:
                                est_iter = ["reparam"]

                            for estimator in est_iter:

                                print("estimator",estimator)

                                self.iteration(config,
                                               c,
                                               batch_size,
                                               keys,
                                               log_dataframes,
                                               data,
                                               i,
                                               mean_step_return,
                                               estimator)

                    if not self.args.fixed_epsilon and self.args.log_batch:

                        # Merge all the log dataframes, and save it in output folder
                        cat_log_df = pd.concat(log_dataframes,keys=keys,axis=1)
                        cat_log_df.to_csv(output_folder+"/attack_logs.csv")

                    self.create_dataframe(data,output_folder)

    def summarize_results(self):

        dfs_global = []
        dfs_attack_global = []

        for year_folder in self.year_folders:

            year_output_folder = os.path.join(self.output_folder, year_folder)
            config_list = self.get_config_folders(year_folder)

            dfs_year = []
            dfs_attack_year = []

            for config in config_list:

                config_output_full = os.path.join(year_output_folder,
                                                  config["output folder"])

                dfs_config = []
                dfs_attack_config = []

                # Only one model per year for attack
                for i in range(1):
                    output_folder = os.path.join(config_output_full,
                                                 config["run folders"][i])

                    df = pd.read_csv(output_folder + "/results.csv")

                    if self.args.log_batch:
                        df_attack = pd.read_csv(output_folder + "/attack_logs.csv",index_col=0,header=[0,1,2])
                        df_attack.index.name = ["steps"]
                        dfs_attack_config.append(df_attack)

                    dfs_config.append(df)

                result = self.merge_dataframes(dfs_config,indexes=self.indexes)
                result.to_csv(config_output_full + "/result.csv", index=True)
                dfs_year.append(result)

                if self.args.log_batch:
                    result_attack = self.merge_values(dfs_attack_config)
                    result_attack.to_csv(config_output_full + "/attack_logs.csv", index=True)
                    dfs_attack_year.append(result_attack)

            result = self.merge_dataframes(dfs_year,indexes=self.indexes)
            result.to_csv(year_output_folder + "/result.csv", index=True)
            dfs_global.append(result)

            if self.args.log_batch:
                result_attack = self.merge_values(dfs_attack_year)
                result_attack.to_csv(year_output_folder + "/attack_logs.csv", index=True)
                dfs_attack_global.append(result_attack)

        result = self.merge_dataframes(dfs_global,indexes=self.indexes)
        result.to_csv(self.output_folder + "/result.csv", index=True)

        if self.args.log_batch:
            result_attack = self.merge_values(dfs_attack_global)
            result_attack.to_csv(self.output_folder + "/attack_logs.csv", index=True)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_epsilon', action="store_true", help='X axis is max perturbation')
    parser.add_argument('--training_folder', default="Attack_debug",help='Folder with training logs')
    parser.add_argument('--print', type=int,default=-1,help='Print loss every .. steps')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples for inference')
    parser.add_argument('--n_iterations', type=int, default=500,
                        help='Number of iterations for attack')
    parser.add_argument('--batch_size',nargs='+', type=int, default=50,
                        help='Batch size for perturbation generation')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for attack')
    parser.add_argument('--c',nargs='+', type=float, default=0.1,
                        help='list of c coefficients (see Carlini et al.)')
    parser.add_argument('--days', type=int, default = 85,help='Number of days batched together')
    parser.add_argument('--output_folder', help='Output folder')
    parser.add_argument('--debug', action="store_true", help='Debug mode')

    # Only for fixed c
    parser.add_argument('--log_batch', action="store_true", help='Mode for visualizing batch effect')
    parser.add_argument('--delta_log',type=int,default=10,help='Interval between points')

    # Only for fixed epsilon
    parser.add_argument('--target',choices=["binary","regression"], help="Attack target")
    parser.add_argument('--confidence', type=float, default=0.95, help='Max perturbation L2 norm')
    parser.add_argument('--max_pert', nargs='+',type=float, default=0.1,help='Max perturbation L2 norm')

    # Only for non conditional
    parser.add_argument('--steps', type=int, default=10, help='Number of steps')
    parser.add_argument('--k', nargs='+', type=int, default=10, help='Number of buy/sell for trading benchmark')

    # Only for conditional
    parser.add_argument('--conditional', action="store_true", help='Conditional mode')
    parser.add_argument('--step_prediction', type=int, default=5, help='Step for prediction')
    parser.add_argument('--step_condition', type=int, default=10, help='Step for condition')
    parser.add_argument('--value_condition', type=float, default=1., help='Value for condition')

    # ARgument for run
    parser.add_argument('--run', action="store_true",help="Run mode")

    # Argument for summarization
    parser.add_argument('--summarize', action="store_true",help="Summarize mode")
    parser.add_argument('--sum_attack_logs', action="store_true", help='Summarize attack logs')

    args = parser.parse_args()

    loop = AttackLoop(args)
    if args.run:
        loop.run()
    if args.summarize:
        loop.summarize_results()


