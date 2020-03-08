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

import os
import json
import pandas as pd
import numpy as np

import sys
sys.path.insert(1, 'common')
import utils
from project_structure import *


class IteratedPredictionLoop():

    def __init__(self,args,out_folder):

        self.folder = os.path.join(training_folder,args.training_folder)
        self.device = utils.choose_device()
        self.debug = args.debug
        self.steps = args.steps
        self.samples = args.samples
        self.batched_days = args.days
        self.k = args.k

        if not isinstance(self.samples,list):
            self.samples = [self.samples]
        if not isinstance(self.k,list):
            self.k = [self.k]

        if not os.path.exists(out_folder):
            os.mkdir(out_folder)


        self.output_folder = os.path.join(out_folder, args.output_folder)
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)


        self.year_folders = self.list_dir(self.folder)

        print(self.year_folders)

    def make_output_folder(self,year_folder):
        year_output_folder = os.path.join(self.output_folder, year_folder)
        if not os.path.exists(year_output_folder):
            os.mkdir(year_output_folder)
        return year_output_folder

    def load_config(self,year_folder,config_folder):
        with open(os.path.join(self.folder,
                               year_folder,
                               config_folder,
                               "parameters.json")) as json_file:
            config = json.load(json_file)
        print(config)

        config_folder_full = os.path.join(self.folder,
                                          year_folder,
                                          config_folder)
        run_folders = self.list_dir(config_folder_full)

        n_reps = len(run_folders)
        print(run_folders)

        return config,config_folder_full,run_folders,n_reps

    def make_folder(self):
        if not os.path.exists(attack_log_folder):
            os.mkdir(attack_log_folder)

        self.folder = attack_log_folder+self.folder_name+"/"
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

    def create_batch_subfolder(self,i,folder):

        subfolder = folder + "batch_" + str(i) + "/"
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)
        return subfolder

    def list_dir(self, folder):

        return [name for name in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, name))]


    def merge_dataframes(self,dfs,indexes=["steps", "names", "prices","samples"]):

        df_with_index = pd.concat([dfs[i].assign(rep=i) for i in range(len(dfs))])
        grouped_df = df_with_index.groupby(indexes)

        mean = grouped_df.mean()
        std = grouped_df.std().add_prefix("std ")
        result = pd.merge(mean, std, left_index=True, right_index=True)
        del result["rep"]
        del result["std rep"]
        return result

    def merge_dataframes_mean(self,dfs,indexes=["steps", "names", "prices","samples"]):

        df_with_index = pd.concat(dfs,keys=[i for i in range(len(dfs))])


        grouped_df = df_with_index.groupby(indexes)

        mean = grouped_df.mean()
        return mean

    def merge_values(self, dfs):

        vals = np.array([df.values for df in dfs])
        mean_vals = np.mean(vals,axis=0)

        df = pd.DataFrame(columns=dfs[0].columns,data=mean_vals)

        return df