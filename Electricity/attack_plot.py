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

import matplotlib

import attack_utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import json
import numpy as np


def plot_batch(original_mu ,original_sigma,
               perturbed_output_mu, perturbed_output_sigma,
               best_c, best_perturbation ,best_distance, labels,
               targets, params, batch_folder):

    batch_size = original_mu.shape[0]


    nrows = 3
    ncols = 3
    size = nrows*ncols

    all_samples = np.arange(batch_size)
    random_sample = np.random.choice(all_samples, size=size, replace=False)


    for mode in ["over","under"]:

        label_plot = labels[random_sample]
        original_mu_chosen = original_mu[random_sample]
        original_sigma_chosen = original_sigma[random_sample]


        x = np.arange(params["test_window"])

        target_index = -2
        for tolerance in range(perturbed_output_mu["over"].shape[0]):
            f = plt.figure(figsize=(20,7), constrained_layout=True)
            ax = f.subplots(nrows, ncols)

            for k0 in range(nrows):

                for k1 in range(ncols):

                    k = nrows * k1 + k0

                    ax[k0][k1].plot(x[:target_index + 1],
                                    np.concatenate([label_plot[k,:params["predict_start"]],
                                    original_mu_chosen[k][:target_index+1]]),
                                    color='r',
                                    linewidth=4)

                    ax[k0][k1].fill_between(x[params["predict_start"]:target_index + 1],
                                            original_mu_chosen[k][:target_index + 1] - \
                                            original_sigma_chosen[k][:target_index + 1],
                                            original_mu_chosen[k][:target_index + 1] + \
                                            original_sigma_chosen[k][:target_index + 1], color='r',
                                            alpha=0.2)


                    mu_chosen = perturbed_output_mu[mode][tolerance][random_sample]
                    sigma_chosen = perturbed_output_sigma[mode][tolerance][random_sample]

                    ax[k0][k1].fill_between(x[params["predict_start"]:target_index+1],
                                       mu_chosen[k][:target_index+1] - \
                                        sigma_chosen[k][:target_index+1],
                                       mu_chosen[k][:target_index+1] + \
                                        sigma_chosen[k][:target_index+1], color='blue',
                                       alpha=0.2)

                    pert = (1 + best_perturbation[mode][tolerance][1:, random_sample])

                    aux = label_plot[k, :params["predict_start"]] * pert[:params["predict_start"] ,k]

                    # Plot adversarial sample 1
                    ax[k0][k1].plot(x[:target_index+1],
                               np.concatenate([aux,mu_chosen[k][:target_index+1]]),
                                    color='blue',
                                    linewidth=2)


                    ax[k0][k1].axvline(params["predict_start"], color='g', linestyle='dashed')

                    ax[k0][k1].set_ylim(ymin=0)
                    ax[k0][k1].grid()

                    if k0 == 2 and k1 == 1:
                        ax[k0][k1].set_xlabel("Hour")

                    if k0 == 1 and k1 == 0:
                        ax[k0][k1].set_ylabel("Electricity consumption")


            str_tol = str(params["tolerance"][tolerance])
            name = mode+'_tolerance_'+str_tol+'.png'
            f.savefig(os.path.join(batch_folder,name))
            plt.close()

folder = os.path.join("attack_logs","attack_results")
batch_folder = os.path.join("attack_logs","attack_results","batch_0")

loader = attack_utils.H5pySaver(batch_folder)

original_mu = {}
original_sigma = {}
best_c = {}
best_perturbation = {}
best_distance = {}
perturbed_output_mu = {}
perturbed_output_sigma = {}
targets = {}
labels = {}

estimator = "reparam"
original_mu = loader.get_from_file(estimator+'_original_mu')
original_sigma = loader.get_from_file(estimator+'_original_sigma')
best_c = loader.get_dict_from_file(estimator+'_best_c')
best_perturbation = loader.get_dict_from_file(estimator+'_best_perturbation')
best_distance = loader.get_dict_from_file(estimator+'_best_distance')
perturbed_output_mu = loader.get_dict_from_file(estimator+'_perturbed_output_mu')
perturbed_output_sigma = loader.get_dict_from_file(estimator+'_perturbed_output_sigma')
targets = loader.get_dict_from_file(estimator+'_targets')
labels = loader.get_from_file(estimator+'_labels')

with open(os.path.join(folder,"params.txt")) as json_file:
    params = json.load(json_file)

plot_batch(original_mu,
           original_sigma,
           perturbed_output_mu,
           perturbed_output_sigma,
           best_c,
           best_perturbation,
           best_distance,
           labels,
           targets,
           params,
           batch_folder)

