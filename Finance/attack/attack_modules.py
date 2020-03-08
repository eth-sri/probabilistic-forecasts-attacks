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

import torch.nn as nn
import torch

import sys
sys.path.insert(0, 'common')
import utils

EPS = 0.000001
DELTA = 0.03
ALPHA = 0.1
N_SAMPLES = 100

class SuperAttackModule(nn.Module):

    def __init__(self,model,args,c,input_):

        super(SuperAttackModule, self).__init__()

        self.model = model
        self.args = args
        self.c = c
        self.input = input_
        self.n_inputs = input_.shape[1]
        self.attack_loss = CustomAttackLoss(c)

        # Initialize perturbation
        self.perturbation = nn.Parameter(torch.zeros(self.input.shape, device=self.model.device))

    def generate_target(self, batch_size, mean, mode,n_per_day,ground_truth=None,steps=None,estimator=None):

        # Generate mean
        with torch.no_grad():
            if estimator is not None and estimator == "deterministic":
                perturbed_output = self.forward_deterministic()
            else:
                perturbed_output = self(n_samples=N_SAMPLES)

        # Average per day
        perturbed_output_per_day = perturbed_output.view(n_per_day,-1)
        mean_output = torch.mean(perturbed_output_per_day,axis=0).unsqueeze(-1)
        mean_output = mean_output.expand(mean_output.shape[0],n_per_day).reshape(-1)

        if mode == "sell":

            target = mean_output - DELTA
        elif mode == "buy":
            target = mean_output + DELTA

        elif mode == "regression":

            y = ground_truth[:,steps-1,0]

            target = mean_output - ALPHA * (y - mean_output)
        else:
            raise Exception("No such mode")

        return target,mean_output

    def set_n_samples(self,n_samples):

        if n_samples is None:
            # Only works with one batch size
            if self.batch_size is None:
                n_samples = self.args.batch_size[0]
            else:
                n_samples = self.batch_size

        return n_samples

    def perturb_input(self,perturbation):

        # Size (sample,batch_size,self.nSteps)
        if perturbation is None:
            perturbed_input = self.input * (1 + self.perturbation)
        else:
            perturbed_input = self.input * (1 + perturbation)

        return perturbed_input

    def common_step(self,perturbed_input):

        # Preprocess input
        returns = self.model.preprocessing(perturbed_input).to(self.model.device)

        # Compute before last state
        irrelevant_output, cell = self.model.get_distrib(returns[:-1])

        return returns,cell



class AttackModule(SuperAttackModule):

    def __init__(self,model,args,c,input_,batch_size=None):

        super(AttackModule,self).__init__(model,args,c,input_)

        self.n_steps = self.args.steps

        self.batch_size = batch_size

    def forward(self,n_samples=None,perturbation=None,std=False,debug=False):

        n_samples = self.set_n_samples(n_samples)

        perturbed_input = self.perturb_input(perturbation)


        batch = self.input.shape[1]

        returns, cell = self.common_step(perturbed_input)

        sum_first_moment = torch.zeros(batch, device=self.model.device)

        if std:
            sum_second_moment = torch.zeros(batch, device=self.model.device)

        for i in range(n_samples):
            outputs, _ = self.model.forward_prediction_sample(returns[-1:],
                                             cell,
                                             self.n_steps,
                                             mode="reparam")

            out = outputs[:, self.args.steps-1].squeeze(1)
            sum_first_moment += out

            if std:
                sum_second_moment += out**2

        if std:
            sem = sum_second_moment - sum_first_moment**2/float(n_samples)

            sem /= (n_samples*(n_samples-1))
            sem = torch.sqrt(sem)

        mean = sum_first_moment/float(n_samples)

        if std:
            return mean,sem

        return mean

    def forward_naive(self):

        n_samples = self.args.batch_size[0]
        batch = self.input.shape[1]

        perturbed_input = self.input * (1 + self.perturbation)

        returns, cell = self.common_step(perturbed_input)

        # Compute estimator
        sum_first_moment = torch.zeros(batch, device=self.model.device)

        outputs = torch.zeros(n_samples,batch,self.n_steps,1,device=self.model.device)

        with torch.no_grad():
            for i in range(n_samples):
                # Store outputs and weights
                outputs[i], _ = self.model.forward_prediction_sample(returns[-1:],
                                             cell,
                                             self.n_steps,
                                             mode="reparam")

                out = outputs[i,:, self.args.steps-1].squeeze(1)
                sum_first_moment += out

            mean = sum_first_moment / float(n_samples)

        # Forward pass on all samples
        aux_estimate = torch.zeros(batch,device=self.model.device)
        for i in range(n_samples):
            log_prob = self.model.forward_log_prob(returns[-1:],
                                                   cell,
                                                   self.args,
                                                   self.n_steps,
                                                   outputs[i],
                                                   conditioned=False)

            aux_estimate += outputs[i, :,self.args.steps-1].squeeze(1)*log_prob

        aux_estimate /= float(n_samples)
        aux_estimate = aux_estimate.sum(0)

        return mean,aux_estimate

    def forward_deterministic(self,perturbation=None):

        perturbed_input = self.perturb_input(perturbation)

        returns = self.model(perturbed_input,"prediction",n_steps=self.args.steps,attack=True)

        return returns[:, self.args.steps-1].squeeze(1)


class ConditionalAttackModule(SuperAttackModule):

    def __init__(self,model,args,c,input_,batch_size=None):

        super(ConditionalAttackModule,self).__init__(model,args,c,input_)

        self.n_steps = max(self.args.step_condition,self.args.step_prediction)

        self.batch_size = batch_size

    def forward(self,n_samples=None,perturbation=None,std=False):

        n_samples = self.set_n_samples(n_samples)

        perturbed_input = self.perturb_input(perturbation)

        batch = self.input.shape[1]

        returns,cell = self.common_step(perturbed_input)


        sum_weights = torch.zeros(batch, device=self.model.device)
        sum_first_moment = torch.zeros(batch, device=self.model.device)

        if std:
            sum_squared_weights = torch.zeros(batch, device=self.model.device)
            sum_second_moment = torch.zeros(batch, device=self.model.device)
            sum_hybrid_moment = torch.zeros(batch, device=self.model.device)


        for i in range(n_samples):
            outputs, weights = self.model.forward_conditioned(returns[-1:],
                                                        cell,
                                                        self.args,
                                                        self.n_steps)

            out = outputs[:, self.args.step_prediction-1].squeeze(1)

            sum_first_moment += out*weights

            if std:
                sum_squared_weights += weights**2
                sum_second_moment += (out*weights)**2
                sum_hybrid_moment += out*(weights**2)

            sum_weights += weights

        if std:
            sem = (sum_second_moment+
                   sum_squared_weights*sum_first_moment**2/sum_weights**2-
                   2*sum_hybrid_moment*sum_first_moment/sum_weights)

            sem /= (n_samples*(n_samples-1))
            sem = torch.sqrt(sem)

        mean = sum_first_moment/sum_weights

        if std:
            return mean,sem

        return mean

    def forward_naive(self):

        n_samples = self.args.batch_size[0]
        batch = self.input.shape[1]

        perturbed_input = self.input * (1 + self.perturbation)

        returns, cell = self.common_step(perturbed_input)

        # Compute estimator
        sum_weights = torch.zeros(batch, device=self.model.device)
        sum_first_moment = torch.zeros(batch, device=self.model.device)

        outputs = torch.zeros(n_samples,batch,self.n_steps-1,1,device=self.model.device)
        weights = torch.zeros(n_samples,batch,device=self.model.device)

        with torch.no_grad():
            for i in range(n_samples):
                # Store outputs and weights
                outputs[i], weights[i] = self.model.forward_conditioned(returns[-1:],
                                                                  cell,
                                                                  self.args,
                                                                  self.n_steps)

                out = outputs[i,:, self.args.step_prediction-1].squeeze(1)
                sum_first_moment += out * weights[i]
                sum_weights += weights[i]

            mean = sum_first_moment / sum_weights

        # Forward pass on all samples
        aux_estimate = torch.zeros(batch,device=self.model.device)
        for i in range(n_samples):
            log_prob = self.model.forward_log_prob(returns[-1:],
                                                   cell,
                                                   self.args,
                                                   self.n_steps,
                                                   outputs[i])



            aux_estimate += weights[i]*outputs[i, :,self.args.step_prediction-1].squeeze(1)*log_prob

        aux_estimate /= sum_weights
        aux_estimate = aux_estimate.sum(0)

        return mean,aux_estimate

class CustomAttackLoss(nn.Module):

    def __init__(self, c):
        super(CustomAttackLoss, self).__init__()
        self.c = c
        self.device = utils.choose_device()

    def forward(self, perturbation, output, target):

        loss_function = nn.MSELoss(reduction="none")
        distance_per_sample = loss_function(output, target)

        distance = distance_per_sample.sum(0)

        zero = torch.zeros(perturbation.shape).to(self.device)
        norm_per_sample = loss_function(perturbation, zero).sum(0).sum(1)
        norm = norm_per_sample.sum(0)

        loss_per_sample = norm_per_sample + self.c * distance_per_sample
        loss = norm + self.c * distance

        return norm_per_sample,distance_per_sample,loss_per_sample,norm,distance,loss


