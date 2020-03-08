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
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import pandas as pd

import attack_utils
import utils
import model.net as net
from attack_utils import AttackLoss
from dataloader import *

logger = logging.getLogger('DeepAR.Eval')

class AttackModule(nn.Module):

    def __init__(self, model, params, c, data,id_batch, v_batch, hidden, cell):

        super(AttackModule, self).__init__()

        self.model = model
        self.params = params
        self.c = c
        self.data = data
        self.id_batch = id_batch
        self.v_batch = v_batch
        self.hidden = hidden
        self.cell = cell
        self.n_inputs = data.shape[1]
        self.attack_loss = AttackLoss(params,c,v_batch)

        # Initialize perturbation
        self.perturbation = nn.Parameter(torch.zeros(self.data.shape[:2], device=self.params.device))

    def generate_target(self,labels,mode):

        if mode == "over":
            target = 1.5*labels
        elif mode == "under":
            target = 0.5*labels
        else:
            raise Exception("No such mode")

        return target

    # Returns mean and std
    def forward(self):

        perturbed_data = torch.zeros(self.data.shape).to(self.params.device)
        perturbed_data[:,:,0] = self.data[:,:,0]  * (1 + self.perturbation)
        perturbed_data[:,:,1:] = self.data[:,:,1:]

        samples, sample_mu, sample_sigma = attack_utils.forward_model(model,
                                                                      perturbed_data,
                                                                      self.id_batch,
                                                                      self.v_batch,
                                                                      self.hidden,
                                                                      self.cell,
                                                                      self.params)

        return samples,sample_mu,sample_sigma

    # Not clear yet how to compute that
    def forward_naive(self):

        perturbed_data = torch.zeros(self.data.shape).to(self.params.device)
        perturbed_data[:, :, 0] = self.data[:, :, 0] * (1 + self.perturbation)
        perturbed_data[:, :, 1:] = self.data[:, :, 1:]

        with torch.no_grad():
            samples, sample_mu,_ = attack_utils.forward_model(model,
                                                              perturbed_data,
                                                              self.id_batch,
                                                              self.v_batch,
                                                              self.hidden,
                                                              self.cell,
                                                              self.params)

        aux_estimate = torch.zeros(self.data.shape[1], device=self.params.device)

        hidden = self.hidden
        cell = self.cell
        for t in range(self.params.test_predict_start):

            mu, sigma, hidden, cell = model(perturbed_data[t].unsqueeze(0),
                                            self.id_batch,
                                            hidden,
                                            cell)

        for i in range(samples.shape[0]):
            sample = samples[i]
            log_prob = attack_utils.forward_log_prob(model,
                                                     sample,
                                                      perturbed_data,
                                                      self.id_batch,
                                                      self.v_batch,
                                                      hidden,
                                                      cell,
                                                      self.params)
            aux_estimate += sample[:,self.params.target] * log_prob

        aux_estimate /= float(samples.shape[0])
        aux_estimate = aux_estimate.sum(0)

        return sample_mu,aux_estimate


class Attack():

    def __init__(self,model, loss_fn, test_loader, params, plot_num):
        self.model = model
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.params = params
        self.plot_num = plot_num

        # Set the model to eval mode
        # Replaced model.eval() to deal with
        # RuntimeError: cudnn RNN backward can only be called in training mode
        # (https: // github.com / pytorch / pytorch / issues / 10006)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

            elif isinstance(module, nn.LSTM):
                module.dropout = 0

        print('model: ', model)

        self.max_pert_len = len(params.tolerance)

    def print(self,i,norm,distance,loss,batch):

        print(i,norm.detach().item()/batch,
              distance.detach().item()/batch,
              loss.detach().item()/batch)

    def project_perturbation(self,attack_module):

        aux = torch.tensor([-1.], device=self.params.device)

        attack_module.perturbation.data = torch.max(attack_module.perturbation.data, aux)


    def attack_step_ours(self, attack_module, optimizer, i, target):

        attack_module.zero_grad()

        _,prediction,_ = attack_module()

        norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, prediction, target)

        # Differentiate loss with respect to input
        loss.backward()

        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)

    def attack_step_naive(self, attack_module, optimizer, i, target):

        attack_module.zero_grad()

        mean, aux_estimate = attack_module.forward_naive()

        aux_estimate.backward()

        # Compute the derivative of the loss with respect to the mean
        mean.requires_grad = True
        attack_module.perturbation.requires_grad = False
        _,_,_,_,_, loss = attack_module.attack_loss(attack_module.perturbation, mean, target)

        # This propagates the gradient to the mean
        loss.backward()

        # Multiply the two, and set it in perturbation
        attack_module.perturbation.grad *= mean.grad[:,self.params.target]

        # Compute the derivative of the loss with respect to the norm
        mean.requires_grad = False
        attack_module.perturbation.requires_grad = True
        norm_per_sample,_,_,norm,distance, loss = \
            attack_module.attack_loss(attack_module.perturbation, mean, target)


        # This propagates the gradient to the norm
        loss.backward()

        # Apply one step of optimizer
        optimizer.step()

        self.project_perturbation(attack_module)

    def attack_batch(self, batch_number, data, id_batch, v_batch, labels, hidden, cell, estimator):

            batch_size = data.shape[1]

            with torch.no_grad():
                _, original_mu, original_sigma = attack_utils.forward_model(model,
                                                                          data,
                                                                          id_batch,
                                                                          v_batch,
                                                                          hidden,
                                                                          cell,
                                                                          self.params)

            shape = (self.max_pert_len,) + data.shape[:2]

            best_perturbation = {"over": np.zeros(shape),
                                 "under": np.zeros(shape)}

            c_shape = (self.max_pert_len, data.shape[1])
            best_c = {"over": np.zeros(c_shape),
                      "under": np.zeros(c_shape)}
            best_distance = {"over": np.full(c_shape, np.inf),
                             "under": np.full(c_shape, np.inf)}

            out_shape = (self.max_pert_len,)+ original_mu.shape
            perturbed_output_mu = {"over": np.zeros(out_shape),
                                   "under": np.zeros(out_shape)}
            perturbed_output_sigma = {"over": np.zeros(out_shape),
                                      "under": np.zeros(out_shape)}

            modes = ["under","over"]
            targets = {}

            lines = []

            for mode in modes:

                print("mode",mode)

                # Loop on values of c to find successful attack with minimum perturbation

                for i in range(0, len(self.params.c), self.params.batch_c):

                    bound = min(self.params.batch_c,len(self.params.c)-i)

                    batched_data = data.repeat(1,bound,1)

                    batched_id_batch = id_batch.repeat(1,bound)
                    batched_v_batch = v_batch.repeat(bound,1)
                    batched_labels = labels.repeat(bound)
                    batched_hidden = hidden.repeat(1,bound,1)
                    batched_cell = cell.repeat(1,bound,1)

                    batched_c = torch.cat([self.params.c[i+j]*\
                                           torch.ones(batch_size,device=self.params.device)\
                                           for j in range(bound)],dim = 0)

                    # Update the lines
                    attack_module = AttackModule(self.model,
                                                 self.params,
                                                 batched_c,
                                                 batched_data,
                                                 batched_id_batch,
                                                 batched_v_batch,
                                                 batched_hidden,
                                                 batched_cell)

                    batched_target = attack_module.generate_target(batched_labels,mode)

                    optimizer = optim.Adam([attack_module.perturbation], lr=self.params.learning_rate)

                    # Iterate steps
                    for k in range(self.params.n_iterations):

                        if estimator == "reparam":
                            self.attack_step_ours(attack_module, optimizer, k, batched_target)
                        elif estimator == "score":
                            self.attack_step_naive(attack_module, optimizer, k, batched_target)
                        else:
                            raise Exception("No such estimator")

                    # Evaluate the attack
                    # Run full number of samples on perturbed input to obtain perturbed output
                    with torch.no_grad():
                        _,batched_perturbed_output,_ = attack_module()

                        # Unbatch c to run everything from this
                        for j in range(bound):

                            c = self.params.c[i+j]

                            left = batch_size*j
                            right = batch_size*(j+1)

                            target = batched_target[left:right]
                            targets[mode] = target

                            perturbed_output = batched_perturbed_output[left:right]
                            v_batch = batched_v_batch[left:right]

                            loss = attack_utils.AttackLoss(self.params,c,v_batch)

                            norm_per_sample, distance_per_sample, loss_per_sample, norm, distance, loss = \
                                loss(attack_module.perturbation[:,left:right],
                                                          perturbed_output,
                                                          target)

                            # Find
                            numpy_norm = np.sqrt(utils.convert_from_tensor(norm_per_sample))
                            numpy_distance = utils.convert_from_tensor(distance_per_sample)
                            numpy_perturbation = utils.convert_from_tensor(
                                attack_module.perturbation.data[:,left:right])

                            for l in range(self.max_pert_len):

                                indexes_best_c = np.logical_and(numpy_norm <= self.params.tolerance[l],
                                                                numpy_distance < best_distance[mode][l])

                                best_perturbation[mode][l][:, indexes_best_c] = \
                                    numpy_perturbation[:, indexes_best_c]
                                best_distance[mode][l, indexes_best_c] = \
                                    numpy_distance[indexes_best_c]
                                best_c[mode][l, indexes_best_c] = c

                            # Save norm and distance for c plot
                            mean_numpy_norm = np.mean(numpy_norm)
                            mean_distance = np.mean(np.sqrt(numpy_distance))

                            lines.append([batch_number,estimator,mode,c,mean_numpy_norm,mean_distance])

                with torch.no_grad():

                    # Update the lines
                    attack_module = AttackModule(self.model,
                                                 self.params,
                                                 0,
                                                 data,
                                                 id_batch,
                                                 v_batch,
                                                 hidden,
                                                 cell)

                    for l in range(self.max_pert_len):

                        attack_module.perturbation.data = \
                            torch.tensor(best_perturbation[mode][l],
                                         device=self.params.device).float()
                        _,aux1,aux2 = attack_module()

                        perturbed_output_mu[mode][l] = aux1.cpu().numpy()
                        perturbed_output_sigma[mode][l] = aux2.cpu().numpy()


            return original_mu,original_sigma,best_c, best_perturbation, \
                   best_distance, perturbed_output_mu, perturbed_output_sigma,\
                   targets,lines

    def attack(self):

        lines = []

        # For each test sample
        # Test_loader:
        # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # id_batch ([batch_size]): one integer denoting the time series id;
        # v ([batch_size, 2]): scaling factor for each window;
        # labels ([batch_size, train_window]): z_{1:T}.
        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(self.test_loader)):

                print("\n")
                print("Batch",i)

                index = v[:,0] > 0
                test_batch = test_batch[index]
                id_batch = id_batch[index]
                v = v[index]
                labels = labels[index]

                # Prepare batch data
                test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)
                id_batch = id_batch.unsqueeze(0).to(params.device)
                v_batch = v.to(torch.float32).to(params.device)
                test_labels = labels.to(torch.float32).to(params.device)[:,params.target]

                batch_size = test_batch.shape[1]
                hidden = model.init_hidden(batch_size)
                cell = model.init_cell(batch_size)

                for estimator in ["score", "reparam"]:

                    print("estimator",estimator)

                    original_mu,original_sigma,best_c,best_perturbation,best_distance,\
                        perturbed_output_mu, perturbed_output_sigma,targets, l = \
                        self.attack_batch(i,test_batch,id_batch,v_batch,
                                          test_labels,hidden,cell,estimator)
                    lines += l

                    # Save results
                    saver = attack_utils.H5pySaver(os.path.join(params.output_folder,"batch_"+str(i)+"/"))

                    saver.save_to_file(original_mu, estimator + '_original_mu')
                    saver.save_to_file(original_sigma, estimator + '_original_sigma')
                    saver.save_dict_to_file(best_c, estimator + '_best_c')
                    saver.save_dict_to_file(best_perturbation, estimator + '_best_perturbation')
                    saver.save_dict_to_file(best_distance, estimator + '_best_distance')
                    saver.save_dict_to_file(perturbed_output_mu, estimator + '_perturbed_output_mu')
                    saver.save_dict_to_file(perturbed_output_sigma, estimator + '_perturbed_output_sigma')
                    saver.save_dict_to_file(targets, estimator + '_targets')
                    saver.save_to_file(labels, estimator + '_labels')

        df = pd.DataFrame(lines,columns=["batch","estimator","mode","c","norm","distance"])
        df.to_csv(os.path.join(params.output_folder,"results.csv"))



if __name__ == '__main__':

    params,model_dir,args,data_dir = attack_utils.set_params()

    model = attack_utils.set_cuda(params,logger)

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)

    loss_fn = net.loss_fn

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    attack = Attack(model, loss_fn, test_loader, params, -1,)

    test_metrics = attack.attack()
    save_path = os.path.join(model_dir, 'metrics_test_{}.json'.format(args.restore_file))