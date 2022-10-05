import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from Model import Net
import copy


class Made():
    def __init__(self, args, device, PARAM_PATH = 'Best_params', preset_ordering = False, ordering=[None]):
        self.args = args
        self.model = Net(self.args.nin, self.args.hiddens, self.args.nout, preset_ordering, self.args.num_masks, ordering)
        self.device = device    
        self.model.to(self.device)


        self.num_dist_params = int(self.args.nout/self.args.nin)
        if self.num_dist_params == 1:
            self.conditonal_dist = 'Binary' # Must be 'Binary', 'Gaussian'
            self.loss_fn = nn.BCEWithLogitsLoss( reduction='sum')
        else:
            self.conditonal_dist = 'Gaussian' 
            self.loss_fn = nn.GaussianNLLLoss(reduction='sum',full=True)


        self.param_path = PARAM_PATH
        self.best_params = torch.save(self.model.state_dict(), self.param_path)
        self.opt = torch.optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=45, gamma=0.1)


    def train(self, xtr, xte, max_iter_ = 1000, convergence_criteron = 0.001, early_stop = 30, upto=5):
        epoch = 0
        prev_nll = None 
        prev_test_nll = None
        not_converged = True
        no_improvement_counter = 0

        while not_converged and epoch < max_iter_ and no_improvement_counter < early_stop:
            # New epoch updates
            epoch += 1
            print("epoch %d" % (epoch, ))
            
            
            # Get test and train log likelihoods
            test_nll = self.test(xte, upto) # run only a few batches for approximate test accuracy
            nll = self.run_epoch(xtr, 'train')


            self.scheduler.step()

            # Check for improved validation performance, if yes update best params
            if epoch == 1 or test_nll < prev_test_nll:
                self.best_params = torch.save(self.model.state_dict(), self.param_path)
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Check for convergence
            if epoch > 1:
                if abs(prev_nll - nll) < convergence_criteron:
                    not_converged = False
                else:
                    prev_test_nll = test_nll
                    prev_nll = nll
            else:
                # If it is the first epoch
                prev_test_nll = test_nll
                prev_nll = nll

        if not_converged:
            if epoch == max_iter_:
                print('Max iter reached')
            else:
                print('Early stopping due to no validation improvement')
        else:
            print('Converged')




    def test(self, X, upto):
        return self.run_epoch(X, 'test', upto)


    def sample(self, n_samples=100, pixel_values = True):
        self.model.eval()
        
        order = self.model.ordering
        print('Used ordering: ', order)

        # Empty matrix for samples
        samples = torch.zeros(n_samples, self.model.nin)
                
        # sample the first dimension of each vector
        if self.conditonal_dist == 'Binary':
            samples[:, np.where(order == 0)[0][0]] =  torch.distributions.Bernoulli(torch.rand(n_samples)).sample()
        elif pixel_values:
            # pixel values are fixed between 0 and 1, thus we sample from a uniform (0,1) dist instead of the normal
            samples[:, np.where(order == 0)[0][0]] = torch.rand(n_samples)
        else:
            # If not pixel values, we can use standard gaussian for initial value
            samples[:, np.where(order == 0)[0][0]] = torch.randn(n_samples)
                
        for i in range(self.model.nin):
            # Get which feature is next in AR ordering
            inv_swap = np.where(order == i)[0][0]
            
            # Get output of model given all 0 expect orderings already sampled
            if self.conditonal_dist == 'Binary':
                p = torch.sigmoid(self.model(samples.to(self.device)))
                bernoulli = torch.distributions.Bernoulli(p[:, inv_swap])
                sample_output = bernoulli.sample()
            else:
                # Gaussian Case does not require sigmoid - output is mu, log(sigma^2)
                output  = self.model(samples.to(self.device))
                mu, alpha = torch.chunk(output, 2, dim=1)
                var = torch.exp(alpha)
                sample_output = torch.normal(mu[:, inv_swap], torch.sqrt(var[:, inv_swap]))
                
            samples[:, inv_swap] = sample_output

        return samples

    def select_best_params(self):
        self.model.load_state_dict(torch.load(self.param_path))
        print('Best Parameters loaded')

    def run_epoch(self, x, split, upto=None):
        torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
        self.model.train() if split == 'train' else self.model.eval()
        nsamples = 1 if split == 'train' else self.args.samples
        N,D = x.size()
        B = self.args.batch_size # batch size
        nsteps = N//B if upto is None else min(N//B, upto)
        lossfs = []
        for step in range(nsteps):
            
            # fetch the next batch of data
            xb = Variable(x[step*B:step*B+B])
            
            # get the logits, potentially run the same batch a number of times, resampling each time
            xbhat = torch.zeros((xb.shape[0], xb.shape[1]*self.num_dist_params)).to(self.device)
            for s in range(nsamples):
                # perform order/connectivity-agnostic training by resampling the masks
                if step % self.args.resample_every == 0 or split == 'test': # if in test, cycle masks every time
                    self.model.update_masks()
                    # Print orderings to check order agnostic and connectivity agnostic training
                    #print(self.model.ordering)
                # forward the model
                xbhat += self.model(xb.float())
            xbhat /= nsamples
            
            if self.conditonal_dist == 'Binary':
                # evaluate the binary cross entropy loss = Bernouli NLL
                loss = self.loss_fn(xbhat, xb) / B
            elif self.conditonal_dist == 'Gaussian':
                # evaluate the gaussian negative log-likelihood 
                # We estimate log(sigma^2) to avoid issues with estimated negative var
                mu, alpha = torch.chunk(xbhat, 2, dim=1)
                var = torch.exp(alpha)
                loss = self.loss_fn(mu, xb, var) / B
        
            lossf = loss.data.item()
            lossfs.append(lossf) 
            
            # backward/update
            if split == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            
        print("%s epoch average loss: %f" % (split, np.mean(lossfs)))
    
    
        return np.mean(lossfs)

class arguments:
    def __init__(self, nin, nout, hiddens, num_masks, resample_every, samples, batch_size = 100):
        self.nin = nin
        self.nout = nout
        self.hiddens = hiddens
        self.num_masks = num_masks
        self.resample_every = resample_every
        self.samples = samples
        self.batch_size = batch_size