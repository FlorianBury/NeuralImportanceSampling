import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional



class AffineCouplingLayer(nn.Module): 
    def __init__(self, input_dim, mask, n_layers, hidden_dim, batchnorm = False):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim  = input_dim     # Dimension of the problem
        self.n_layers   = n_layers      # Number of layers
        self.hidden_dim = hidden_dim    # Number of nodes per layer 
        self.batchnorm  = batchnorm     # Apply batch norm on each layer of scale and translation

        # mask to seperate positions that do not change and positions that change.
        # mask[i] = 1 means the ith position does not change.
        self.mask       = mask

        # Layers of scale in affine transformation #
        s_layers = [nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(self.n_layers):
            s_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        s_layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        self.scale_layers = nn.ModuleList(s_layers) 

        # Batch norm for scale #
        if self.batchnorm:
            s_layers_bn = [nn.BatchNorm1d(self.hidden_dim) for _ in range(self.n_layers+1)]        
            s_layers_bn.append(nn.BatchNorm1d(self.input_dim))
            self.scale_layers_batchnorm = nn.ModuleList(s_layers_bn) 

        # Layers of translation in affine transformation #
        t_layers = [nn.Linear(self.input_dim, self.hidden_dim)]
        for i in range(self.n_layers):
            t_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        t_layers.append(nn.Linear(self.hidden_dim, self.input_dim))
        self.translation_layers = nn.ModuleList(t_layers)

        # Batch norm for translation #
        if self.batchnorm:
            t_layers_bn = [nn.BatchNorm1d(self.hidden_dim) for _ in range(self.n_layers+1)] 
            t_layers_bn.append(nn.BatchNorm1d(self.input_dim))
            self.translation_layers_batchnorm = nn.ModuleList(t_layers_bn) 

    def _compute_scale(self, x):
        det = 1
        #print ("======================")
        #print ("scale")
        for i,layer in enumerate(self.scale_layers):
            x = layer(x)
            if self.batchnorm:
                x = self.scale_layers_batchnorm[i](x) # TODO : might need Jacobian
                det *= (self.scale_layers_batchnorm[i].weight / torch.sqrt(self.scale_layers_batchnorm[i].running_var+self.scale_layers_batchnorm[i].eps)).prod() # gamma*(sigma^2 + epsilon)^-1/2
                print (det)
            if i == self.n_layers+1: # output activation is sigmoid
                x = nn.Tanh()(x)
            else:                  # hidden activation is relu
                x = nn.LeakyReLU(0.1)(x)
        return x, det

    def _compute_translation(self, x):
        det = 1
        #print ("======================")
        #print ("translation")
        for i,layer in enumerate(self.translation_layers):
            x = layer(x)
            if self.batchnorm:
                x = self.translation_layers_batchnorm[i](x)
                det *= (self.translation_layers_batchnorm[i].weight / torch.sqrt(self.translation_layers_batchnorm[i].running_var+self.translation_layers_batchnorm[i].eps)).prod() # gamma*(sigma^2 + epsilon)^-1/2
                print (det)
            if i != self.n_layers+1: # No activation on last layer
                x = nn.LeakyReLU(0.1)(x)  # LeakyRelu in hidden
        return x, det

    def forward(self,z):
        # From Z (latent variables) to Y (observed variables)
        zm = z * self.mask
        s, s_det = self._compute_scale(zm)
        t, t_det = self._compute_translation(zm)
        y = zm + (1-self.mask)*(z*torch.exp(s) + t) 
        det = torch.exp(torch.sum(s, 1))*s_det*t_det
            # Triangular matrix : det = product of diagonal elements
            # for x with unchanged positions : identity matrix
            # for x with changed positions : triangular with diagonal with elements of exp(s)

        return y, det

    def backward(self,y):
        # From Y (observed variables) to Z (latent variables) 
        ym = y * self.mask
        s, s_det = self._compute_scale(ym)
        t, t_det = self._compute_translation(ym)

        z = ym + (1-self.mask)*((y - t)*torch.exp(-s))
        det = torch.exp(torch.sum(-s, 1))*s_det*t_det
            # With the inverse transformation : s -> -s

        return z, det

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask, n_couplinglayers, n_hiddenlayers, batchnorm = False):
        super(RealNVP,self).__init__()

        # COncatenate couplig layers #
        modules = []
        modules.append(AffineCouplingLayer(input_dim, mask, n_hiddenlayers, hidden_dim, batchnorm))
        for i in range(1,n_couplinglayers):
            mask = 1-mask # Needs to switch which variables are changed 
            modules.append(AffineCouplingLayer(input_dim, mask, n_hiddenlayers, hidden_dim, batchnorm))
        self.modulelist  = nn.ModuleList(modules)

        # Batch norm #
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(n_couplinglayers)])

    def forward(self,z):
        # From Z (latent variables) to Y (observed variables)
        y = z
        det_tot = 1
        for i,module in enumerate(self.modulelist):
            y, det = module(y)
            det_tot = det_tot * det
            #if self.batchnorm:
            #    y = self.batchnorm_layers[i](y) 
            #    det_tot *= (self.batchnorm_layers[i].weight / torch.sqrt(self.batchnorm_layers[i].running_var+self.batchnorm_layers[i].eps)).prod() # gamma*(sigma^2 + epsilon)^-1/2

        return y , det_tot

    def backward(self,y):
        # From Y (observed variables) to Z (latent variables) 
        z = y
        det_tot = 1
        for module in reversed(self.modulelist):
            z, det = module.backward(z)
            det_tot = det_tot * det

        return z, det_tot
