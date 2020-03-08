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
import torch.optim as optim
import numpy as np
import math
import scipy

import sys
sys.path.insert(0, 'common')
import utils as utils
from attack_modules import CustomAttackLoss, AttackModule, ConditionalAttackModule

EPS = 0.000001


class Attack():

    def log_batch(self,attack_module,i,log,batch_delta_log,target):

        if self.args.log_batch and not i % batch_delta_log:

            with torch.no_grad():
                perturbed_output = attack_module(n_samples=self.args.samples,debug=True)

                _,_,_, norm, distance, loss = \
                    attack_module.attack_loss(attack_module.perturbation, perturbed_output, target)

                index = int(i / batch_delta_log)
                log[index][0] = norm.detach().item()
                log[index][1] = distance.detach().item()
                log[index][2] = loss.detach().item()

    def print_iteration(self, i, norm, distance, loss):

        if self.args.print > 0 and i % self.args.print == 0:
            print("Iteration: ", i)
            print(norm.detach().item(),distance.detach().item(),loss.detach().item())

    # Do projection to ensure that the price remains positive
    def project_perturbation(self,attack_module):

        aux = torch.tensor([-1 + EPS], device=attack_module.model.device)

        attack_module.perturbation.data = torch.max(attack_module.perturbation.data,aux)

    def print_grad(self,name,attack_module):

        print(name, attack_module.perturbation.grad.detach().cpu().numpy()[-3:, :3])

    def attack_step_reparam(self,attack_module,optimizer,i,target,
                         log=None,batch_delta_log=0):

        attack_module.zero_grad()

        # with torch.autograd.detect_anomaly():
        prediction = attack_module()

        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, prediction, target)

        self.print_iteration(i,norm,distance,loss)

        # Differentiate loss with respect to input
        loss.backward()

        # Log the loss at each timestep
        self.log_batch(attack_module,i,log,batch_delta_log,target)

        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)


    def attack_step_score(self, attack_module, optimizer, i, target):

        attack_module.zero_grad()

        mean,aux_estimate = attack_module.forward_naive()

        aux_estimate.backward()

        # Compute the derivative of the loss with respect to the mean
        mean.requires_grad = True
        attack_module.perturbation.requires_grad = False
        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, mean, target)

        # This propagates the gradient to the mean
        loss.backward()

        # Compute grad of aux estimate

        # Multiply the two, and set it in perturbation
        attack_module.perturbation.grad *= mean.grad.unsqueeze(-1)

        # Compute the derivative of the loss with respect to the norm
        mean.requires_grad = False
        attack_module.perturbation.requires_grad = True
        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, mean, target)

        # This propagates the gradient to the norm
        loss.backward()

        self.print_iteration(i,norm,distance,loss)

        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)

    def attack_step_deterministic(self,attack_module,optimizer,i,target):

        attack_module.zero_grad()

        # with torch.autograd.detect_anomaly():
        prediction = attack_module.forward_deterministic()

        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, prediction, target)

        self.print_iteration(i,norm,distance,loss)

        # Differentiate loss with respect to input
        loss.backward()

        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)

class FixedCAttack(Attack):

    def __init__(self,model,args,mean_step_return,c,batch_size,log_batch_size):

        super(FixedCAttack,self).__init__()
        self.args = args
        self.model = model
        self.mean_step_return = mean_step_return
        self.c = c
        self.batch_size = batch_size

        self.modes = ["buy","sell"]

        if self.args.log_batch:
            # Will only work as expected if self.batch_size | self.args.delta_log
            assert(self.args.delta_log % self.batch_size == 0)
            assert(self.args.n_iterations % self.batch_size == 0)
            self.batch_delta_log = int(self.args.delta_log / self.batch_size)
            self.n_iterations = int(self.args.n_iterations/self.batch_size)
        else:
            self.batch_delta_log = 0
            self.n_iterations = self.args.n_iterations

        self.log_batch_size = log_batch_size

    def attack(self,x,n_per_day):

        n_inputs = x.shape[1]

        final_perturbation = {}
        final_norm = {}
        final_distance = {}

        if self.args.log_batch:
            logs = {}

        for mode in self.modes:

            attack_module = AttackModule(self.model, self.args, self.c, x,batch_size=self.batch_size)

            target,_ = attack_module.generate_target(n_inputs,
                                                     self.mean_step_return,
                                                     mode,
                                                     n_per_day)
            optimizer = optim.RMSprop([attack_module.perturbation], lr=self.args.learning_rate)

            if self.args.log_batch:
                log = np.zeros((self.log_batch_size,3))
            else:
                log = None

            # Iterate steps
            for i in range(self.n_iterations):

                self.attack_step_reparam(attack_module,
                                      optimizer,
                                      i,
                                      target,
                                      log=log,
                                      batch_delta_log=self.batch_delta_log)

            # Evaluate the attack
            # Run full number of samples on perturbed input to obtain perturbed output
            with torch.no_grad():
                perturbed_output = attack_module(n_samples=self.args.samples)

                norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
                    attack_module.attack_loss(attack_module.perturbation, perturbed_output, target)

                final_perturbation[mode] = utils.convert_from_tensor(attack_module.perturbation.detach())
                final_norm[mode] = utils.convert_from_tensor(norm_per_sample)
                final_distance[mode] = utils.convert_from_tensor(distance_per_sample)

                if self.args.log_batch:
                    logs[mode] = log

        if self.args.log_batch:
            return final_perturbation,final_norm,final_distance,logs
        else:
            return final_perturbation,final_norm,final_distance

class FixedEpsilonAttack(Attack):

    def __init__(self,model,args,mean,estimator):

        super(FixedEpsilonAttack,self).__init__()

        self.model = model
        self.args = args
        self.mean = mean
        self.max_pert_len = len(self.args.max_pert)
        self.estimator = estimator

        if self.args.target == "binary":
            self.target_type = "binary"
        else:
            self.target_type = "regression"
            # Check that not a conditional
            assert(not self.args.conditional)

    def compute_metrics(self,mode,output,sem,mean_output):

        h = scipy.stats.t.ppf(self.args.confidence, self.args.samples)

        if mode == "buy":
            bound = output - h*sem
            test = bound >= mean_output

        elif mode == "sell":
            bound = output + h*sem
            test = bound <= mean_output

        number = np.count_nonzero(utils.convert_from_tensor(test))

        return number

    def attack(self,input_,n_per_day,ground_truth=None):

        if self.target_type == "binary":
            best_perturbation = {"buy":np.zeros((self.max_pert_len,)+input_.shape),
                                 "sell":np.zeros((self.max_pert_len,)+input_.shape)}

            c_shape = (self.max_pert_len,input_.shape[1])
            best_c = {"buy":np.zeros(c_shape),
                      "sell":np.zeros(c_shape)}
            best_distance = {"buy":np.full(c_shape,np.inf),
                             "sell":np.full(c_shape,np.inf)}
            percentage = {}
        else:
            best_perturbation = {"regression":np.zeros((self.max_pert_len,) + input_.shape)}

            c_shape = (self.max_pert_len, input_.shape[1])
            best_c = {"regression":np.zeros(c_shape)}
            best_distance = {"regression":np.full(c_shape, np.inf)}
            perturbed_outputs = []

        if self.target_type == "binary":
            modes = ["buy","sell"]
        else:
            modes = ["regression"]

        for mode in modes:
            # Loop on values of c to find successful attack with minimum perturbation

            for i in range(len(self.args.c)):

                c = self.args.c[i]

                # Create attack module with parameters
                if self.args.conditional:
                    attack_module = ConditionalAttackModule(self.model,self.args,c,input_)
                else:
                    attack_module = AttackModule(self.model,self.args,c,input_)

                target,mean_output = attack_module.generate_target(input_.shape[1],
                                                       self.mean,
                                                       mode,
                                                       n_per_day,
                                                       ground_truth=ground_truth,
                                                       steps=self.args.steps,
                                                       estimator=self.estimator)
                optimizer = optim.RMSprop([attack_module.perturbation], lr=self.args.learning_rate)

                # Iterate steps
                for i in range(self.args.n_iterations):

                    if self.estimator == "reparam":
                        self.attack_step_reparam(attack_module,optimizer,i,target)
                    elif self.estimator == "score":
                        self.attack_step_score(attack_module, optimizer, i, target)
                    else:
                        raise Exception("No such estimator")

                # Evaluate the attack
                # Run full number of samples on perturbed input to obtain perturbed output
                with torch.no_grad():

                    if self.estimator == "deterministic":

                        perturbed_output = attack_module.forward_deterministic()
                    else:

                        perturbed_output = attack_module(n_samples=self.args.samples)

                    norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
                        attack_module.attack_loss(attack_module.perturbation, perturbed_output, target)

                    # Find
                    numpy_norm = np.sqrt(utils.convert_from_tensor(norm_per_sample))
                    numpy_distance = utils.convert_from_tensor(distance_per_sample)
                    numpy_perturbation = utils.convert_from_tensor(attack_module.perturbation.data)

                    for l in range(self.max_pert_len):
                        indexes_best_c = np.logical_and(numpy_norm <= self.args.max_pert[l]-0.00001,
                                                        numpy_distance < best_distance[mode][l])

                        best_perturbation[mode][l][:,indexes_best_c] = \
                            numpy_perturbation[:,indexes_best_c]
                        best_distance[mode][l,indexes_best_c] =\
                            numpy_distance[indexes_best_c]
                        best_c[mode][l,indexes_best_c] = c

            with torch.no_grad():
                if self.target_type == "binary":
                    percentage[mode] = []
                    for l in range(self.max_pert_len):

                        # Check if 95% confidence interval is in "buy" or "sell"
                        attack_module.perturbation.data = \
                            torch.tensor(best_perturbation[mode][l],
                                        device=attack_module.model.device).float()

                        if self.estimator == "deterministic":
                            perturbed_output = attack_module.forward_deterministic()
                            sem = 0.
                        else:
                            perturbed_output,sem = attack_module(n_samples=self.args.samples,std=True)

                        metrics = self.compute_metrics(mode,perturbed_output,sem,mean_output)
                        percentage[mode].append(metrics)
                else:

                    for l in range(self.max_pert_len):
                        # Check if 95% confidence interval is in "buy" or "sell"
                        attack_module.perturbation.data = \
                            torch.tensor(best_perturbation[mode][l],
                                         device=attack_module.model.device).float()

                        if self.estimator == "deterministic":
                            out = attack_module.forward_deterministic()
                        else:
                            out = attack_module(n_samples=self.args.samples, std=False)
                        perturbed_outputs.append(out)

        if self.target_type == "binary":
            return best_c,best_perturbation,best_distance,percentage
        else:
            return perturbed_outputs








