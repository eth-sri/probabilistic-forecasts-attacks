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
import math
from itertools import compress
import pandas as pd
import os
import json

import sys
sys.path.insert(1, 'common')
from project_structure import *
from data_downloader import DataDownloader

class Dataset:

    def __init__(self,year,input_length,output_length,batch_size):

        # Fraction of validation samples (out of training samples)
        self.fraction_val = 0.15
        self.fraction_train = 1 - self.fraction_val
        self.name = "Year_"+str(year)
        self.folder = datasets_folder+self.name+"/"

        self.training_folder = self.folder+"training/"
        self.val_folder = self.folder+"val/"
        self.test_folder = self.folder+"test/"

        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length+output_length

        d = DataDownloader()
        self.ticker_list = d.read_constituents()

        print("\n")
        print("Test year",year)

        start_test_year,end_test_year = self.read_data(year)

        # Create folder if not existing
        for folder in [self.folder,self.training_folder,self.val_folder,self.test_folder]:
            if not os.path.exists(folder):
                os.mkdir(folder)

        self.meta = {}

        self.get_data(start_test_year,end_test_year,batch_size)

    def read_data(self,year):

        self.data = pd.read_hdf(raw_data_hdf)
        year_indexes = np.where(self.data.index.year==year)
        start_index = year_indexes[0][0]-750
        end_index = year_indexes[0][-1]+self.total_length+50
        self.data = self.data[start_index:end_index]

        # Remove days where the stock market was closed
        check = self.data.isnull().all(axis=1).values
        print("Days where stock market was closed",check.sum())
        self.data = self.data[~check]

        # Find new start_index for test year
        year_indexes = np.where(self.data.index.year == year)
        start_test_year = year_indexes[0][0]
        end_test_year = year_indexes[0][-1]+1
        self.end_training = math.ceil(self.fraction_train * start_test_year +
                                      self.fraction_val * self.total_length -
                                      self.fraction_train * self.output_length +
                                      2*self.fraction_train - 1)

        # Remove companies that where not present at a given period
        self.ticker_list = self.filter_ticker_list()

        return start_test_year,end_test_year

    def filter_ticker_list(self):

        values = [self.data[ticker]['Close'].values for ticker in self.ticker_list]
        check_nan = [not ((v != v).any()) for v in values]
        tickers = list(compress(self.ticker_list, check_nan))

        # Remove NVR (anomaly in 1993)
        tickers.remove("NVR")

        return tickers

    # Gets data at index start for given ticker
    def get_at(self,index,input_length,output_length,ticker,prep_y=True):

        data = self.data[ticker]['Close'].values
        length = input_length+output_length

        # Get data
        x = data[index:index+input_length]

        if prep_y:
            y = (data[index+input_length:index+length]/
                 data[index+input_length-1:index+length-1])
            # Normalize y with mean and std
            y = (y - self.mean)/self.std
        else:
            y = data[index+input_length:index+length]

        return x,y

    # Function to get minibatches
    # mode is "training" or "validation" or "test"
    def get_minibatches(self,mode,start_index,end_index,batch_size):

        n_samples = end_index - start_index - self.total_length + 1

        number_batches = 0
        x_batch = []
        y_batch = []
        # Double loop on ticker and index
        for n in range(n_samples):
            for ticker in self.ticker_list:

                start = start_index+n

                x,y = self.get_at(start,self.input_length,self.output_length,ticker)

                x_batch.append(x)
                y_batch.append(y)

                if len(x_batch) == batch_size:
                    # Save batch to files
                    self.save_batch_to_file(x_batch,y_batch,number_batches,mode)
                    number_batches += 1
                    # Reinit batch
                    x_batch = []
                    y_batch = []

        # Last batch
        if len(x_batch) > 0:
            # Save batch to files
            self.save_batch_to_file(x_batch, y_batch, number_batches, mode)
            number_batches += 1
        return number_batches

    # Function to get test per day batches (batch size is not fixed)
    def get_test_per_day(self,start_index,end_index,name):

        n_samples = end_index - start_index - self.total_length + 1

        prep_y = True

        # Here we always group per day
        for n in range(n_samples):
            x_batch = []
            y_batch = []
            for ticker in self.ticker_list:

                start = start_index+n

                x,y = self.get_at(start,self.input_length,self.output_length,ticker,prep_y)

                x_batch.append(x)
                y_batch.append(y)

            self.save_batch_to_file(x_batch,y_batch,n,name)

        # DEBUG
        return n_samples

    def compute_means(self,start_test_year):

        # Get data
        # print("Tickers",tickers)
        vals = [self.data[ticker]['Close'][:start_test_year].values for ticker in self.ticker_list]
        nums = [np.copy(a[1:]) for a in vals]
        returns = [nums[i]/vals[i][:-1] for i in range(len(vals))]
        returns = np.array(returns)

        self.mean = np.mean(returns)
        self.std = np.std(returns)
        print("Mean return",self.mean)
        print("Std return",self.std)

    # Gets training, validation and test set
    def get_data(self,start_test_year,end_test_year,batch_size):

        # Compute means
        self.compute_means(start_test_year)

        # Get batches
        n_train = self.get_minibatches("training",
                                       0,
                                       self.end_training,
                                       batch_size)
        n_val = self.get_minibatches("validation",
                                     self.end_training - self.input_length,
                                     start_test_year,
                                     batch_size)
        n_test = self.get_test_per_day(start_test_year - self.input_length,
                                       end_test_year + self.output_length - 1,
                                       "test")
        # Get meta info
        self.meta["n train batches"]=int(n_train)
        self.meta["n val batches"]=int(n_val)
        self.meta["n test batches"]=int(n_test)
        self.meta["mean"]=self.mean
        self.meta["std"]=self.std
        with open(self.folder+"meta.json","w") as f:
            f.write(json.dumps(self.meta,indent=4))

    def save_batch_to_file(self,x_batch,y_batch,batch_number,mode):

        if mode == "training":
            folder = self.training_folder
        elif mode == "test":
            folder = self.test_folder
        elif mode == "validation":
            folder = self.val_folder

        x_batch = np.transpose(np.array(x_batch))
        x_dataframe = pd.DataFrame(data=x_batch)
        x_file = folder+"x_"+str(batch_number)+".csv"
        x_dataframe.to_csv(x_file)

        y_dataframe = pd.DataFrame(data=y_batch)
        y_file = folder+"y_"+str(batch_number)+".csv"
        y_dataframe.to_csv(y_file)

def generate_dataset(args):

    for year in range(args.start_year,args.end_year+1):
        Dataset(year,args.input_length+1,args.output_length,args.batch_size)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year', type=int, default = 1993,help='First year of the period')
    parser.add_argument('--end_year', type=int, default = 2000, help='End year of the period')
    parser.add_argument('--input_length', type=int, default = 240, help='Output length')
    parser.add_argument('--output_length', type=int, default = 50, help='Output length')
    parser.add_argument('--batch_size', type=int, default = 2048, help='Output length')
    args = parser.parse_args()

    generate_dataset(args)



