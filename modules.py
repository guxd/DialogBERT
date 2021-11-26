# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import math
import random
from queue import PriorityQueue
import operator
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules

class MLP(nn.Module):
    def __init__(self, input_size, arch, output_size, activation=nn.ReLU(), batch_norm=True, init_w=0.02, discriminator=False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_w= init_w
        
        if type(arch) == int: arch= str(arch) # simple integer as hidden size
        layer_sizes = [input_size] + [int(x) for x in arch.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and not(discriminator and i==0):# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        if x.dim()==3:
            sz1, sz2, sz3 = x.size()
            x = x.view(sz1*sz2, sz3)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        if x.dim()==3:
            x = x.view(sz1, sz2, -1)
        return x

    def init_weights(self):
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, self.init_w)
                layer.bias.data.fill_(0)
            except:
                pass
  

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network. [Bishop, 1994]. Adopted from https://github.com/tonyduan/mdn
    References: 
        http://cbonnett.github.io/MDN.html
        https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """
    def __init__(self, dim_in, dim_out, n_components):
        super().__init__()
        self.n_components = n_components
        
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ELU(),
            nn.Linear(dim_in, n_components)
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ELU(),
            nn.Linear(dim_in, 2 * dim_out * n_components),
        )

    def forward(self, x):
        pi_logits = self.pi_network(x)
        pi = torch.distributions.OneHotCategorical(logits=pi_logits)
        
        normal_params = self.normal_network(x)
        mean, sd = torch.split(normal_params, normal_params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
        normal = torch.distributions.Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))
        
        return pi, normal

    def loss(self, pi, normal, y):
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss.mean()

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples

