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

import pyro
import torch

EPS = 0.00001


# Used in mdn model
def sample_from_distrib(distrib ,batch_size ,j):

    locs = distrib[0]
    scales = distrib[1]
    comp_prob = distrib[2]

    # Batch
    distrib_0 = pyro.distributions.Categorical(comp_prob)
    comp = pyro.sample("comp_step_{}".format(j), distrib_0)

    # Sample from this component
    selected_locs = torch.gather(locs, 1, comp.unsqueeze(-1).expand(batch_size, 3))[:, 0]
    selected_scales = torch.gather(scales, 1, comp.unsqueeze(-1).expand(batch_size, 3))[:, 0]

    sample_returns = pyro.sample("ret_step_{}".format(j),
                                 pyro.distributions.Normal(selected_locs,
                                                           selected_scales)). \
        unsqueeze(0).unsqueeze(2)

    return sample_returns


# Used in mdn model
def sample_from_distrib_reparametrized(distrib ,batch_size ,j,batch_first=False):

    locs = distrib[0]
    scales = distrib[1]
    comp_prob = distrib[2]

    assert(locs.shape[-1] == 1)
    selected_locs = locs[:,0]
    selected_scales = scales[ :, 0]

    device = locs.get_device()
    if device >= 0:
        mean = torch.zeros(selected_locs.shape,device=device)
        std = torch.ones(selected_scales.shape,device=device)
    else:
        mean = torch.zeros(selected_locs.shape)
        std = torch.ones(selected_scales.shape)

    distrib = pyro.distributions.Normal(mean,std)
    epsilon = pyro.sample("ret_step_{}".format(j),distrib)

    sample_returns = selected_locs + selected_scales*epsilon

    if batch_first:
        return sample_returns.unsqueeze(1).unsqueeze(2)
    return sample_returns.unsqueeze(0).unsqueeze(2)


def sample_from_joint_distrib(joint_distrib,dev,mean):

    device = joint_distrib.get_device()

    batch_size = joint_distrib.shape[1]
    seq_len = joint_distrib.shape[3]
    sample = torch.zeros(batch_size,seq_len,1).to(device)

    return_product = torch.ones(batch_size,1).to(device)
    for i in range(seq_len):
         ret = sample_from_distrib(joint_distrib[:,:,:,i],batch_size,i)\
             .squeeze(0)
         rescaled_ret = ret * dev + mean
         return_product *= rescaled_ret
         sample[:, i] = return_product


    return sample,None