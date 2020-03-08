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
sys.path.insert(1, 'common')
from project_structure import *

import pandas as pd
import json
import torch
import numpy as np

# Class to read from a datset folder
class DataReader:

    def __init__(self,year,device):

        self.folder = datasets_folder+"Year_"+str(year)

        if self.folder[-1] != "/":
            self.folder += "/"
        self.training_folder = self.folder+"training/"
        self.test_folder = self.folder+"test/"
        self.val_folder = self.folder+"val/"

        # Read meta-data
        json_file = self.folder+"meta.json"
        with open(json_file) as f:
            data = json.load(f)
            self.meta = data
            self.n_train_batches = data["n train batches"]
            self.n_val_batches = data["n val batches"]
            self.n_test_batches = data["n test batches"]

        self.device = device

    def get_batch(self,index,folder,cpu=False,cumulative=False):

        x_batch_file = folder+"x_"+str(index)+".csv"
        y_batch_file = folder+"y_"+str(index)+".csv"

        x_batch = pd.read_csv(x_batch_file).values[:,1:]

        y_batch = pd.read_csv(y_batch_file).values[:,1:]

        # RESHAPING
        x = torch.from_numpy(x_batch[:,:,np.newaxis]).float()
        y = torch.from_numpy(y_batch[:,:,np.newaxis]).float()
        if not cpu:
            x = x.to(self.device)
            y = y.to(self.device)
        if cumulative:
            y = torch.cumprod(y*self.meta["std"]+self.meta["mean"],dim=1)
        return x,y

    def get_training_batch(self,index):
        return self.get_batch(index,self.training_folder)

    def get_val_batch(self,index):
        return self.get_batch(index,self.val_folder)

    def get_test_batch(self,index,cumulative=False):
        return self.get_batch(index,self.test_folder,cumulative=cumulative)


