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
import pandas as pd
import time
import torch

from toolz import interleave

import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'common')

import predictions as pr
import prediction_generator as prg
from iterated_prediction_loop import IteratedPredictionLoop
from model import GroundTruthModel
from model import ClimatologicalModel
from project_structure import *
import utils


class EvalLoop(IteratedPredictionLoop):

    def __init__(self,args):

        super(EvalLoop,self).__init__(args,eval_folder)

        if(not isinstance(self.steps,list)):
            self.steps = [self.steps]

        self.nsteps = len(args.steps)

        self.rps = args.rps
        if self.rps:
            self.n_bins = args.n_bins
            print("RPS mode")
            print("N bins",self.n_bins)

            self.predictions, self.binary_predictions = \
                pr.construct_predictions_list(args.steps,max(self.n_bins,1))

            self.certify = args.certify
            self.radius,self.cdf = pr.get_cdf_thresholds()

            if self.certify:
                print("Radius",self.radius)
                print("Cdf",self.cdf)


    def log_meta(self,time,folder):
        # Log meta file with experiments parameters
        stats = {"Time per run (seconds)":time,"N samples":self.samples}
        with open(folder + "/meta.json", "w") as f:
            f.write(json.dumps(stats,indent=4))


    def get_config_folders(self,year_folder):
        config_folders = self.list_dir(os.path.join(self.folder, year_folder))

        # Duplicate the MDN configs
        config_list = []
        for c in config_folders:
            config,folder_full,run_folders,n_reps = self.load_config(year_folder,c)
            if(config["model"]["type"]=="mdn"):
                config_list.append({"n reps": n_reps,
                                    "training folder": folder_full,
                                    "output folder": "mdn_ours",
                                    "run folders": run_folders,
                                    "config": config})
            elif(config["model"]["type"]=="lstm"):
                config_list.append({"n reps":n_reps,
                                    "training folder": folder_full,
                                    "output folder": "lstm",
                                    "run folders": run_folders,
                                    "config": config})
            elif config["model"]["type"]=="tcn":
                config_list.append({"n reps": n_reps,
                                    "training folder": folder_full,
                                    "output folder": "tcn",
                                    "run folders": run_folders,
                                    "config": config})
            else:
                raise Exception("No such model")

        # Iterate over config
        return config_list

    # Results has shape (repetition,nsteps)
    def get_entries(self,results,config):

        entries = np.zeros((self.nsteps,6))

        if (config["model"]["type"] == "mdn"):
            entries[:,0] = 1
        else:
            entries[:,0] = 0

        entries[:,1] = self.steps

        # Average/Std

        metrics = np.array(results)
        means = np.mean(metrics, axis=0)
        stds = np.std(metrics, axis=0)

        entries[:,2:4] = means
        entries[:,4:6] = stds
        return entries

    def result_to_dataframe(self,result):

        predictions_list = [ self.predictions["steps"],
                  self.predictions["names"],
                  self.predictions["prices"]]

        predictions_list = zip(*predictions_list)

        product = [(x[0],x[1],x[2],y) for x in predictions_list for y in self.samples]

        indexes = pd.MultiIndex.from_tuples(product,names=["steps","names","prices","samples"])

        columns = ["val"]+\
                  ["bin "+str(i) for i in range(self.n_bins)]+\
                  ["RPS"]

        data = result.transpose(1,0,2).reshape(-1,result.shape[2])

        dataframe_for_run = pd.DataFrame(index=indexes,
                                         columns=columns,
                                         data=data)

        dataframe_for_run.sort_index(inplace=True)

        return dataframe_for_run

    def binary_results_to_dataframe(self,result):

        if self.certify:
            len_binary = 4+len(self.cdf)
        else:
            len_binary = 4


        predictions_list = [self.binary_predictions["steps"],
                            self.binary_predictions["names"],
                            self.binary_predictions["prices"]]

        predictions_list = zip(*predictions_list)

        product = [(x[0], x[1], x[2], y) for x in predictions_list for y in self.samples]

        indexes = pd.MultiIndex.from_tuples(product, names=["steps", "names", "prices", "samples"])

        columns = [ "val", "class 0", "class 1", "RPS"]
        if self.certify:
            columns += ["radius " + str(r) for r in self.radius]

        data = result.transpose(1, 0, 2).reshape(-1, result.shape[2])

        dataframe_for_run = pd.DataFrame(index=indexes,
                                         columns=columns,
                                         data=data)

        dataframe_for_run.sort_index(inplace=True)

        return dataframe_for_run

    def trading_results_to_dataframe(self,r):

        iterables = [self.steps,self.k,self.samples]
        indexes = pd.MultiIndex.from_product(iterables, names=["steps","k","samples"])
        data = r.transpose(2,1,0).reshape(-1)

        columns = ["result"]

        dataframe_for_run = pd.DataFrame(index=indexes,
                                         columns=columns,
                                         data=data)

        dataframe_for_run.sort_index(inplace=True)

        return dataframe_for_run

    def compute_pr(self,model):
        model.cdf = self.cdf
        model.log_samples = self.samples
        pr_gen = prg.Prediction_Generator(model,
                                          self.steps,
                                          self.k,
                                          self.batched_days,
                                          self.samples,
                                          rps = self.rps,
                                          predictions = self.predictions,
                                          binary_predictions = self.binary_predictions,
                                          n_bins = self.n_bins,
                                          debug = self.debug)
        rgt,rgt_binary,trading = pr_gen.compute_predictions()
        return rgt,rgt_binary,trading

    def merge_dataframes_inner(self,dfs_inner, dfs_outer, config_output_full,
                               prefix,name,indexes=["steps", "names", "prices","samples"]):

        result = self.merge_dataframes(dfs_inner,indexes=indexes).add_prefix(prefix)
        result.to_csv(config_output_full + name, index=True)
        dfs_outer.append(result)

    def merge_dataframes_outer(self,dfs_outer,dfs_outer_outer,year_output_folder,name):

            result = pd.concat(dfs_outer,axis=1)[list(interleave(dfs_outer))]
            result.to_csv(year_output_folder + name, index=True)
            dfs_outer_outer.append(result)

    def merge_dataframes_outer_outer(self,dfs,name,indexes=["steps", "names", "prices","samples"]):

        result = self.merge_dataframes_mean(dfs,indexes=indexes)
        result.to_csv(self.output_folder + name, index=True)

    def save_results(self,algo_name,result,result_binary,trading,year_output_folder):

        result_dataframe = self.result_to_dataframe(result)
        binary_result_dataframe = self.binary_results_to_dataframe(result_binary)
        trading_result_dataframe = self.trading_results_to_dataframe(trading)

        folder = os.path.join(year_output_folder, algo_name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        result_dataframe.to_csv(folder + "/results.csv")
        result_dataframe = result_dataframe.add_prefix(algo_name+" ")
        binary_result_dataframe.to_csv(folder + "/binary_results.csv")
        binary_result_dataframe = binary_result_dataframe.add_prefix(algo_name + " ")
        trading_result_dataframe.to_csv(folder + "/trading_results.csv")
        trading_result_dataframe = trading_result_dataframe.add_prefix(algo_name + " ")

        return result_dataframe,binary_result_dataframe,trading_result_dataframe


    def ground_truth_predictor(self,year_output_folder):

        model = GroundTruthModel(year_output_folder)
        rgt,rgt_binary,trading = self.compute_pr(model)

        gt_df,gt_binary_df,gt_tr_df = self.save_results("ground_truth",
                                               rgt,
                                               rgt_binary,
                                               trading,
                                               year_output_folder)

        model_clim = ClimatologicalModel(year_output_folder,rgt[-1],rgt_binary[-1])
        rc,rc_binary,trading = self.compute_pr(model_clim)

        rc_df, rc_binary_df,rc_tr_df = self.save_results("climatological",
                                                rc,
                                                rc_binary,
                                                trading,
                                                year_output_folder)

        return gt_df,gt_binary_df,gt_tr_df,rc_df,rc_binary_df,rc_tr_df

    def run(self):

        for year_folder in self.year_folders:

            start_time = time.time()

            year_output_folder = self.make_output_folder(year_folder)
            config_list = self.get_config_folders(year_folder)

            for config in config_list:

                print("\n")
                print("Model", config["output folder"])

                for i in range(config["n reps"]):

                    print("Repetition",i)

                    # print("Repetition",i)
                    run_folder = os.path.join(config["training folder"],
                                              config["run folders"][i])
                    model = torch.load(os.path.join(run_folder, 'model.pt'),
                                       map_location=self.device)

                    # Update parameters for model
                    model.device = utils.choose_device()
                    model.datareader.device = utils.choose_device()
                    model.n_samples = max(self.samples)
                    model.log_samples = self.samples

                    config_output_full = os.path.join(year_output_folder,
                                                 config["output folder"])
                    if not os.path.exists(config_output_full):
                        os.mkdir(config_output_full)
                    output_folder = os.path.join(config_output_full,
                                                 config["run folders"][i])
                    if not os.path.exists(output_folder):
                        os.mkdir(output_folder)

                    if self.rps:
                        pr_gen = prg.Prediction_Generator(model,
                                                          self.steps,
                                                          self.k,
                                                          self.batched_days,
                                                          self.samples,
                                                          rps = self.rps,
                                                          predictions = self.predictions,
                                                          binary_predictions = self.binary_predictions,
                                                          n_bins = self.n_bins,
                                                          debug = self.debug)

                        res, binary_res, trading_results = pr_gen.compute_predictions()

                        dataframe = self.result_to_dataframe(res)
                        dataframe.to_csv(output_folder + "/results.csv")

                        dataframe_binary = self.binary_results_to_dataframe(binary_res)
                        dataframe_binary.to_csv(output_folder + "/binary_results.csv")

                    else:

                        pr_gen = prg.Prediction_Generator(model,
                                                          self.steps,
                                                          self.k,
                                                          self.batched_days,
                                                          self.samples,
                                                          rps=self.rps,
                                                          debug=self.debug)


                        trading_results = pr_gen.compute_predictions()

                    dataframe_trading = self.trading_results_to_dataframe(trading_results)
                    dataframe_trading.to_csv(output_folder + "/trading_results.csv")

            # Compute mean time per config and rep, and round
            end_time = time.time()
            elapsed_time = round((end_time - start_time) /
                                 float(config["n reps"] * len(config_list) ), 2)
            self.log_meta(elapsed_time, year_output_folder)


    def summarize_results(self):

        if self.rps:
            dfs_outer_outer = []
            binary_dfs_outer_outer = []

        tr_dfs_outer_outer = []

        for year_folder in self.year_folders:

            year_output_folder = os.path.join(self.output_folder, year_folder)
            config_list = self.get_config_folders(year_folder)

            if self.rps:
                dfs_outer = []
                binary_dfs_outer = []
            tr_dfs_outer = []

            if self.rps:
                gtr,gtbr,gttr,cr,cbr,ctr = self.ground_truth_predictor(year_output_folder)
                dfs_outer += [gtr,cr]
                binary_dfs_outer += [gtbr,cbr]
                tr_dfs_outer += [gttr,ctr]

            for config in config_list:

                config_output_full = os.path.join(year_output_folder,
                                                  config["output folder"])

                if self.rps:
                    dfs_inner = []
                    binary_dfs_inner = []

                tr_dfs_inner = []

                for i in range(config["n reps"]):

                    output_folder = os.path.join(config_output_full,
                                                 config["run folders"][i])

                    if self.rps:
                        df = pd.read_csv(output_folder+"/results.csv")
                        binary_df = pd.read_csv(output_folder + "/binary_results.csv")
                        dfs_inner.append(df)
                        binary_dfs_inner.append(binary_df)

                    tr_df = pd.read_csv(output_folder + "/trading_results.csv")
                    tr_dfs_inner.append(tr_df)

                # Create a panel fo dataframes
                prefix = config["output folder"]+" "

                if self.rps:

                    self.merge_dataframes_inner(dfs_inner,dfs_outer,
                                                config_output_full,prefix,"/result.csv")
                    self.merge_dataframes_inner(binary_dfs_inner, binary_dfs_outer,
                                                config_output_full, prefix, "/binary_result.csv")

                self.merge_dataframes_inner(tr_dfs_inner, tr_dfs_outer,
                                            config_output_full, prefix, "/trading_result.csv",
                                            indexes=["steps","k","samples"])

            if self.rps:

                self.merge_dataframes_outer(dfs_outer,dfs_outer_outer,year_output_folder,
                                            "/result.csv")
                self.merge_dataframes_outer(binary_dfs_outer, binary_dfs_outer_outer,
                                            year_output_folder,"/binary_result.csv")

            self.merge_dataframes_outer(tr_dfs_outer, tr_dfs_outer_outer,
                                        year_output_folder,"/trading_result.csv")

        if self.rps:
            self.merge_dataframes_outer_outer(dfs_outer_outer,"/result.csv")
            self.merge_dataframes_outer_outer(binary_dfs_outer_outer,"/binary_result.csv")

        self.merge_dataframes_outer_outer(tr_dfs_outer_outer,"/trading_result.csv",
                                          indexes=["steps","k","samples"])



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_folder', help='Folder with training logs')
    parser.add_argument('--steps', type=int,nargs='+',default=[1, 5, 10], help='Number of iterated steps')
    parser.add_argument('--samples', nargs='+',type=int, help='Number of samples for inference')
    parser.add_argument('--days', type=int, help='Number of days batched together')
    parser.add_argument('--k', nargs='+',type=int,help='k for trading strategy')
    parser.add_argument('--output_folder', help='Output folder')
    parser.add_argument('--debug', action="store_true", help='Debug mode')
    parser.add_argument('--rps', action="store_true", help='Rps mode')
    parser.add_argument('--run', action="store_true", help='Run')
    parser.add_argument('--summarize', action="store_true", help='Summarize')

    # For binning mode
    parser.add_argument('--n_bins',default=0, type=int, help='Number of bins')
    parser.add_argument('--certify', action="store_true", help='Certify mode')

    args = parser.parse_args()

    loop = EvalLoop(args)
    if args.run:
        print("\nRUNNING EXPERIMENT")
        loop.run()
    if args.summarize:
        print("\nSUMMARIZING LOGS")
        loop.summarize_results()