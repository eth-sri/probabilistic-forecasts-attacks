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

import h5py
import torch.nn as nn
import torch

import logging
import argparse
import os
import json

import model.net as net
import utils

EPS = 1e-5

class AttackLoss(nn.Module):

    def __init__(self, params, c, v_batch):
        super(AttackLoss, self).__init__()
        self.c = c
        self.device = params.device
        self.v_batch = v_batch
        self.params = params

    # perturbation has shape (nSteps,)
    # output has shape (nSteps,batch_size,output_dim)
    # for the moment, target has shape (batch_size,output_dim)
    def forward(self, perturbation, output, target):


        output = output[:,self.params.target] / self.v_batch[:,0]

        target_normalized = target / self.v_batch[:,0]

        loss_function = nn.MSELoss(reduction="none")
        distance_per_sample = loss_function(output, target_normalized)

        distance = distance_per_sample.sum(0)

        zero = torch.zeros(perturbation.shape).to(self.device)
        norm_per_sample = loss_function(perturbation, zero).sum(0)

        norm = norm_per_sample.sum(0)

        loss_per_sample = norm_per_sample + self.c * distance_per_sample
        loss = loss_per_sample.sum(0)

        return norm_per_sample,distance_per_sample,loss_per_sample,norm,distance,loss


def forward_model(model,data,id_batch,v_batch,hidden,cell,params):

    for t in range(params.test_predict_start):
        # if z_t is missing, replace it by output mu from the last time step
        zero_index = (data[t, :, 0] == 0)
        if t > 0 and torch.sum(zero_index) > 0:
            data[t, zero_index, 0] = mu[zero_index]

        mu, sigma, hidden, cell = model(data[t].unsqueeze(0), id_batch, hidden, cell)


    samples, sample_mu, sample_sigma = model.test(data,
                                                  v_batch,
                                                  id_batch,
                                                  hidden,
                                                  cell,
                                                  sampling=True,
                                                  n_samples=params.batch_size)
    return samples,sample_mu,sample_sigma

def forward_log_prob(model,sample,data,id_batch,v_batch,hidden,cell,params):

    log_prob = model.forward_log_prob(data,
                          sample,
                          v_batch,
                          id_batch,
                          hidden,
                          cell)
    return log_prob

def set_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='elect', help='Name of the dataset')
    parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
    parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
    parser.add_argument('--relative-metrics', action='store_true',
                        help='Whether to normalize the metrics by label scales')
    parser.add_argument('--restore-file', default='best',
                        help='Optional, name of the file in --model_dir containing weights to reload before \
                        training')  # 'best' or 'epoch_#'
    parser.add_argument('--output_folder',  help='Output folder for plots')


    # Attack parameters
    parser.add_argument('--c', nargs='+', type=float, default=[0.01, 0.1, 1, 10, 100],
                        help='list of c coefficients (see Carlini et al.)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', nargs='+', type=int, default=50,
                        help='Batch size for perturbation generation')
    parser.add_argument('--n_iterations', type=int, default=1000,
                        help='Number of iterations for attack')
    parser.add_argument('--target', type=int, default=-7,
                        help='Attacking output time')
    parser.add_argument('--tolerance', nargs='+', type=float, default=[0.01, 0.1, 1],
                        help='Max perturbation L2 norm')

    parser.add_argument('--debug', action="store_true", help='Debug mode')

    # Batching
    parser.add_argument('--batch_c', type=int, default=6, help='Number of c values batched together')

    # Load the parameters
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)

    params = utils.Params(json_path)

    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    params.c = args.c
    params.n_iterations = args.n_iterations
    params.tolerance = args.tolerance
    params.batch_size = args.batch_size
    params.learning_rate = args.lr
    params.output_folder = os.path.join("attack_logs",args.output_folder)
    params.batch_c = args.batch_c
    params.target = args.target

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    with open(os.path.join(params.output_folder, "params.txt"), 'w') as param_file:
        json.dump(params.dict, param_file)

    return params,model_dir,args,data_dir

def set_cuda(params,logger):
    cuda_exist = torch.cuda.is_available()  # use GPU is available

    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    return model

class H5pySaver():

    def __init__(self,folder):

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.subfolder = os.path.join(folder,"raw_data/")

        if not os.path.exists(self.subfolder):
            os.makedirs(self.subfolder)

    def save_to_file(self, data, name):

        file = os.path.join(self.subfolder,name+'.h5')
        with h5py.File(file,'w') as hf:

            if isinstance(data,torch.Tensor):
                data = data.data.cpu().numpy()

            hf.create_dataset(name, data=data)

    def save_dict_to_file(self, data, name):

        for mode in ['over','under']:
            self.save_to_file(data[mode],name+"_"+mode)

    def get_from_file(self, name):

        file = os.path.join(self.subfolder, name + '.h5')
        with h5py.File(file, 'r') as hf:
            return hf[name][:]

    def get_dict_from_file(self,name):

        data = {}

        for mode in ['over', 'under']:
            data[mode] = self.get_from_file(name + "_" + mode)

        return data