import os
import sys
import torch
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt 


#def Gaussian(x,alpha,mu,n):
#    den = 1/(alpha*math.sqrt(n))**n
#    exp = np.exp(-np.power(x-mu,2).sum(axis=1)/alpha**2)
#    return exp/den
#
#def Camel(x,alpha,n):
#    den = 0.5/(alpha*math.sqrt(n))**n
#    mu1 = np.ones(x.shape[1])/3
#    mu2 = np.ones(x.shape[1])*2/3
#    exp1 = np.exp(-np.power(x-mu1,2).sum(axis=1)/alpha**2)
#    exp2 = np.exp(-np.power(x-mu2,2).sum(axis=1)/alpha**2)
#    return (exp1+exp2)/den

def Gaussian(x,alpha,n):
    mu = torch.tensor([0.5]*n)
    den = 1/(alpha*math.sqrt(n))**n
    exp = torch.exp(-(x-mu).pow(2).sum(1)/alpha**2)
    return exp/den

def Camel(x,alpha,n):
    den = 0.5/(alpha*math.sqrt(n))**n
    mu1 = torch.tensor([1/3]*n)
    mu2 = torch.tensor([2/3]*n)
    exp1 = torch.exp(-(x-mu1).pow(2).sum(1)/alpha**2)
    exp2 = torch.exp(-(x-mu2).pow(2).sum(1)/alpha**2)
    return (exp1+exp2)/den

def compute_U1(z,alpha,dim):
    mask = torch.ones_like(z)
    mask[:, 1] = 0.0

    U = (0.5*((torch.norm(z, dim = -1) - 2)/0.4)**2 - \
             torch.sum(mask*torch.log(torch.exp(-0.5*((z - 2)/0.6)**2) +
                                      torch.exp(-0.5*((z + 2)/0.6)**2)), -1))
    return torch.exp(-U)

def compute_U2(z,alpha,dim):
    w1 = torch.sin(2*math.pi*z[:,0]/4)
    U = 0.5*((z[:,1] - w1)/0.4)**2
    return torch.exp(-U)

#def compute_U3(z,alpha,dim):
#    w1 = torch.sin(2*math.pi*z[:,0]/4)
#    w2 = 3*torch.exp(-0.5*((z[:,0] - 1)/0.6)**2)
#    U = -torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w2)/0    .35)**2))
#    return torch.exp(-U)

#def compute_U4(z,alpha,dim):
#    w1 = torch.sin(2*math.pi*z[:,0]/4)
#    w3 = 3*torch.sigmoid((z[:,0] - 1)/0.3)
#    U = -torch.log(torch.exp(-0.5*((z[:,1] - w1)/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1 + w3)/0    .35)**2))
#    return torch.exp(-U)




#dim = 2
#Npoints = 100
#X,Y = torch.meshgrid(torch.linspace(0,1,Npoints),torch.linspace(0,1,Npoints))
#x = torch.cat((X.reshape(-1,1),Y.reshape(-1,1)),1)
#alpha = 0.20
##Z = Gaussian(x,alpha,dim)
#Z = Camel(x,alpha,dim)
#Z = Z.reshape(Npoints,Npoints)
#
#fig = plt.figure(figsize=(6,6))
#
#plt.contourf(X,Y,Z,10)
#plt.show()
#plt.close()

