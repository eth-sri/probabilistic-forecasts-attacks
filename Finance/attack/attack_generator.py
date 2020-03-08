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

import torch
import numpy as np
import math

import sys
sys.path.insert(0, 'common')
import attack_modules,attacks
from project_structure import INPUT_LENGTH
import utils


class Attack:

    def get_x_y(self,i,bound):

        x_cat = []
        y_cat = []
        for j in range(bound):
            x, y = self.model.datareader.get_test_batch(i + j, cumulative=True)
            x_cat.append(x)
            y_cat.append(y)

        n_per_day = x_cat[0].shape[1]

        x_cat = torch.cat(x_cat, dim=1)
        y_cat = torch.cat(y_cat, dim=0)

        return x_cat,y_cat,n_per_day


class FixedCAttackGenerator(Attack):

    def __init__(self,model,args,mean_step_return,c,batch_size,batched_days=None):

        self.model = model
        self.args = args
        self.mean_step_return = mean_step_return
        self.c = c
        self.batch_size = batch_size

        # Not none only if log_batch
        if batched_days != None:
            self.batched_days = batched_days
        else:
            self.batched_days = self.args.days

        if self.args.log_batch:
            self.log_batch_size = math.ceil(self.args.n_iterations/float(self.args.delta_log))
        else:
            self.log_batch_size = None

    def compute_predictions(self):

        n_days = self.model.datareader.n_test_batches

        start_days = 0
        if self.args.debug:
            start_days = 120
            n_days = start_days+self.batched_days

        mean_mse = 0.
        mean_distance = 0.
        mean_abs_per_step = np.zeros(INPUT_LENGTH)
        if self.args.log_batch:
            mean_log_batch = np.zeros((self.log_batch_size,3))
        n_batches = 0

        for i in range(start_days, n_days,self.batched_days):

            n_batches += 1

            bound = min(self.batched_days, n_days - i)

            x_cat, y_cat, n_per_day = self.get_x_y(i, bound)

            fixed_c_attack = attacks.FixedCAttack(self.model,
                                        self.args,
                                        self.mean_step_return,
                                        self.c,
                                        self.batch_size,
                                        self.log_batch_size)

            if self.args.log_batch:
                perturbation,norm,distance,logs= fixed_c_attack.attack(x_cat,n_per_day)
            else:
                perturbation, norm, distance = fixed_c_attack.attack(x_cat,n_per_day)

            # Average perturbation per time step
            for mode in ["buy","sell"]:
                mean_abs_per_step += np.mean(np.mean(np.abs(
                    utils.convert_from_tensor(perturbation[mode])),axis=1),axis=1)
                mean_mse += np.mean(np.sqrt(norm[mode]))
                mean_distance += np.mean(np.sqrt(distance[mode]))

                if self.args.log_batch:
                    mean_log_batch += logs[mode]

        mean_mse /= 2.*n_batches
        mean_distance /= 2*n_batches
        if self.args.log_batch:
            mean_log_batch /= 2*n_batches

        mean_abs_per_step = list(utils.convert_from_tensor(mean_abs_per_step))

        entry = [mean_mse,mean_distance]+mean_abs_per_step
        if self.args.log_batch:
            return entry,mean_log_batch
        else:
            return entry


class FixedEpsilonAttackGenerator(Attack):

        def __init__(self,model,args,mean,estimator):

            self.model = model
            self.args = args

            self.mean = mean

            self.attack = attacks.FixedEpsilonAttack(model, args, mean, estimator)

            self.args.delta_log = 10
            self.max_pert_len = len(self.args.max_pert)
            if self.args.target == "regression":
                self.k_len = len(self.args.k)
            self.estimator = estimator

        def compute_predictions(self):

            n_days = self.model.datareader.n_test_batches

            start_days = 0
            if self.args.debug:
                n_days = start_days + self.args.days
            n_batches = 0

            number_buy = np.zeros(self.max_pert_len)
            number_sell = np.zeros(self.max_pert_len)
            total_number = 0.

            if self.args.target == "regression":
                mean_perturbed = np.zeros((self.k_len,self.max_pert_len))

            for i in range(start_days, n_days, self.args.days):

                n_batches += 1
                bound = min(self.args.days, n_days - i)

                x_cat,y_cat,n_per_day = self.get_x_y(i,bound)

                if self.args.target == "binary":
                    best_c,best_perturbation,best_distance,percentage = \
                        self.attack.attack(x_cat,n_per_day)

                    for l in range(self.max_pert_len):
                        for mode in ["buy","sell"]:
                            aux = np.sqrt((best_perturbation[mode][l]**2).sum(0).reshape(-1))

                            test_indexes = np.logical_or( aux < 0,aux > self.args.max_pert[l])
                            if np.any(test_indexes):
                                print("Max perturbation",self.args.max_pert[l])
                                print(aux[test_indexes])

                            assert(np.all(np.logical_and(0 <= aux,aux <= self.args.max_pert[l])))

                    number_buy += np.array(percentage["buy"])
                    number_sell += np.array(percentage["sell"])

                    total_number += x_cat.shape[1]
                else:

                    perturbed_output = self.attack.attack(x_cat,n_per_day,ground_truth=y_cat)
                    for l in range(self.max_pert_len):

                        for j in range(bound):

                            left = n_per_day * j
                            right = n_per_day * (j + 1)
                            perturbed_output_day_j = perturbed_output[l][left:right]

                            y = y_cat[:, self.args.steps - 1].view(-1).cpu().numpy()
                            y_day_j = y[left:right]

                            perturbed_output_day_j = utils.convert_from_tensor(perturbed_output_day_j)

                            for m in range(self.k_len):
                                mean_perturbed[m,l] += utils.get_returns(perturbed_output_day_j,
                                                                         y_day_j,
                                                                         self.args.k[m])

            if self.args.target == "binary":
                per_buy = number_buy / total_number
                per_sell = number_sell / total_number

                return {"buy":list(per_buy),"sell":list(per_sell)}
            else:

                mean_perturbed /= float(n_days - start_days)
                return list(mean_perturbed)

