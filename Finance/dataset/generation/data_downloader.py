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
import yfinance as yf

class DataDownloader():

    def read_constituents(self):

        # Get list of S&P constituents to download all data
        # From github.com/datasets/s-and-p-500-companies-financials, updated from Wikipedia
        constituents_file = raw_data_folder + "constituents.csv"
        constituents_df = pd.read_csv(constituents_file)
        constituents_list = list(constituents_df['Symbol'].to_numpy())

        return constituents_list

    def download_data(self):

        self.ticker_list = self.read_constituents()

        # Use yfinance to get historical data
        data = yf.download(tickers=self.ticker_list,
                           start="1990-01-01",
                           end="2019-10-01",
                           group_by="ticker")
        data.to_hdf(raw_data_hdf,key="data")

if __name__=="__main__":
    a = DataDownloader()
    a.download_data()
    print("Successfully downloaded data")