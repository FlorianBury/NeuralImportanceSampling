import os
import sys 
import math
import numpy as np
import torch
import torch.nn as nn
import argparse


from visualize import visualize
from RealNVP import RealNVP
from functions import *

torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Integration2D:
    def __init__(self,epochs,batch_size,lr,hidden_dim,n_coupling_layers,n_hidden_layers,save_plt_interval,batchnorm,plot_dir_name):
        self.epochs             = epochs
        self.batch_size         = batch_size
        self.lr                 = lr
        self.hidden_dim         = hidden_dim
        self.n_coupling_layers  = n_coupling_layers
        self.n_hidden_layers    = n_hidden_layers
        self.save_plt_interval  = save_plt_interval
        self.batchnorm          = batchnorm
        self.plot_dir_name      = plot_dir_name
        self.method1_I          = []
        self.method1_sig2       = []
        self.method2_I          = []
        self.method2_sig2       = []
        self.method3_I          = []
        self.method3_sig2       = []
        self.bins2D           = None

        self._run()

    def _defineModel(self):
        mask = torch.tensor([0,1])
        self.model = RealNVP(input_dim          = 2,
                             hidden_dim         = self.hidden_dim,
                             mask               = mask,
                             n_couplinglayers   = self.n_coupling_layers,
                             n_hiddenlayers     = self.n_hidden_layers,
                             batchnorm          = self.batchnorm)

        # Training #
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.prior_z = torch.distributions.uniform.Uniform(torch.tensor([0.0,0.0]), torch.tensor([1.0,1.0]))
        self.lossfunction = nn.MSELoss()

        # Function to integrate #
        self.alpha = 0.2
        self.function  = Camel 
        #self.analytic_integral = 2*(0.5*(math.erf(1/(3*self.alpha))+math.erf(2/(3*self.alpha))))**2
        #self.function  = Gaussian
        #analytic_integral = math.erf(1/(2*self.alpha))**2
        #self.function = compute_U2

        # Visualization plot #
        self.visObject = visualize('Plots/'+self.plot_dir_name)
        #self.visObject.AddAnalyticIntegral(self.analytic_integral)

        self.grid_x1 , self.grid_x2 = torch.meshgrid(torch.linspace(0,1,100),torch.linspace(0,1,100))
        self.grid = torch.cat([self.grid_x1.reshape(-1,1),self.grid_x2.reshape(-1,1)],axis=1)
        func_out = self.function(self.grid,self.alpha,2).reshape(100,100)
        self.visObject.AddContour(self.grid_x1,self.grid_x2,func_out,"Target function : "+self.function.__name__)


    #print (model)
    #for param in model.parameters():
    #  print(param.data)

    def _run(self):
        print ("="*80)
        print ("New model with name %s"%self.plot_dir_name)
        
        # Make model #
        self._defineModel()

        # Loop over epochs #
        for epoch in range(1, self.epochs + 1):
            self._sample(epoch)
            self._train(epoch)

    def _train(self,epoch):
        self.model.train()
        # Generate data #
        z = self.prior_z.sample((self.batch_size,))
        # Reinitialize optimizer #
        self.optimizer.zero_grad()
        # Process through network # 
        x, det = self.model(z)
        # Process through function #
        y = self.function(x,self.alpha,2)
        # Gradient descent #
        #loss = self.lossfunction(y,det)
        loss = -torch.log(y/det).mean()
        #loss = torch.mean(-torch.log(det) - torch.log(y))
        loss.backward()
        cur_loss = loss.item()
        self.optimizer.step()
        print ("Train epoch %d/%d : Loss : %0.10f"%(epoch,self.epochs,cur_loss/z.shape[0]))

    def _sample(self,epoch):
        self.model.eval()
        with torch.no_grad():
            # Process #
            N = 1000
            z = self.prior_z.sample((N,))
            x, det = self.model(z) 
            #print ('mean',x.mean(axis=0))
            #print ('std ',x.std(axis=0))
            #print ('min ',x.min(axis=0)[0])
            #print ('max ',x.max(axis=0)[0])
            #print ('det',det)


            # Evaluate integral #
            y = self.function(x,self.alpha,2) # = f(xn), det = p(xn)
            f = y
            p = 1#det

            # First method : Lepage, A NEW ALGORITHM FOR ADAPTIVE MULTIDIMENSIONAL INTEGRATION (1976)
            S1 = ((f/p).sum()/N).item()
            S2 = ((f/p).pow(2).sum()/N).item()

            self.method1_I.append(S1)
            self.method1_sig2.append((S2-S1**2)/(N-1))

            # v1 #
            method1_Ibar_v1 = 0
            method1_sig2bar_v1 = 0
            method1_chi2perDof_v1 = 0
            for i in range(len(self.method1_I)):
                method1_sig2bar_v1 += 1/self.method1_sig2[i]
            method1_sig2bar_v1 = 1/method1_sig2bar_v1
            for i in range(len(self.method1_I)):
                method1_Ibar_v1 += method1_sig2bar_v1+self.method1_I[i]/self.method1_sig2[i]
            for i in range(len(self.method1_I)):
                method1_chi2perDof_v1 += (self.method1_I[i]-method1_Ibar_v1)**2/self.method1_sig2[i]
            if len(self.method1_I) > 1 :
                method1_chi2perDof_v1 /= (len(self.method1_I)-1)

            self.visObject.AddCurves(x     = epoch,
                                     x_err = 0,
                                     title = "Method 1 : Lepage v1",
                                     dict_val = {r'$I$': [self.method1_I[-1],self.method1_sig2[-1]],
                                                 r'$\bar{I}$': [method1_Ibar_v1,method1_sig2bar_v1]})
                                                 #r'$\chi^2 per dof$':[method1_chi2perDof_v1,0]})
                
            # v2 #
            method1_Ibar_v2 = 0
            method1_sig2bar_v2 = 0
            method1_chi2perDof_v2 = 0
            sum_I2_over_sig2 = 0
            for i in range(len(self.method1_I)):
                method1_Ibar_v2 += self.method1_I[i]*(self.method1_I[i]**2/self.method1_sig2[i])
                sum_I2_over_sig2 += self.method1_I[i]**2/self.method1_sig2[i]
            method1_Ibar_v2 /= sum_I2_over_sig2
            method1_sig2bar_v2 = method1_Ibar_v2**2/sum_I2_over_sig2
            for i in range(len(self.method1_I)):
                method1_chi2perDof_v2 += ((self.method1_I[i]-method1_Ibar_v2)**2/method1_Ibar_v2**2) * (self.method1_I[i]**2/self.method1_sig2[i])
            if len(self.method1_I) > 1 :
                method1_chi2perDof_v2 /= (len(self.method1_I)-1)

            self.visObject.AddCurves(x     = epoch,
                                     x_err = 0,
                                     title = "Method 1 : Lepage v2",
                                     dict_val = {r'$I$': [self.method1_I[-1],self.method1_sig2[-1]],
                                                 r'$\bar{I}$': [method1_Ibar_v2,method1_sig2bar_v2]})
                                                 #r'$\chi^2/dof$':[method1_chi2perDof_v2,0]})

            print ("Method 1 : I = %0.5f +/- %0.5f"%(self.method1_I[-1],self.method1_sig2[-1]))
            print ("\tV1 : Ibar = %0.5f +/- %0.5f (chi2/dof = %0.5f)"%(method1_Ibar_v1,method1_sig2bar_v1,method1_chi2perDof_v1))
            print ("\tV2 : Ibar = %0.5f +/- %0.5f (chi2/dof = %0.5f)"%(method1_Ibar_v2,method1_sig2bar_v2,method1_chi2perDof_v2))

            # Second method : Lepage, VEGAS - An Adaptive Multi-dimensional Integration Program (1980)


            # Third method : Weinzierl, Introduction to Monte Carlo methods (2020)

            ##fop = y/det     # = f(xn)/p(xn)
            #fop = y     # = f(xn)/p(xn)
            #Ej  = fop.sum()/N
            #Sj2 = fop.pow(2).sum()/N-Ej**2

            #self.Ej.append(Ej.item())
            #self.Sj2.append(Sj2.item())
            #self.Nj.append(N)

            #w = 0
            #E = 0
            #for i in range(len(self.Ej)):
            #    E += self.Nj[i]*self.Ej[i]/self.Sj2[i]
            #    w += self.Nj[i]/self.Sj2[i]
            #E /= w

            #chi2dof = 0
            #if len(self.Ej)>1:
            #    for i in range(len(self.Ej)):
            #        chi2dof += (self.Ej[i]-E)**2/self.Sj2[i]
            #    chi2dof /= (len(self.Ej)-1)

            #S1 = (y).sum()/N
            #S2 = (y.pow(2)).sum()/N
            #sig2 = (S2-S1**2)/N
            #self.I.append(S1)
            #self.sigma2.append(sig2)

            #Ibar = 0
            #Ibar_v2 = 0
            #asum = 0
            #asum_v2 = 0
            #den = 0
            #for i in range(len(self.I)):
            #    asum += 1/self.sigma2[i]
            #    b = self.I[i]**2/self.sigma2[i]
            #    Ibar_v2 +=  self.I[i]*b
            #    den += b
            #Ibar_v2 /= den
            #sigbar2_v2 = Ibar_v2/den
            #sigbar2 = 1/asum
            #for i in range(len(self.I)):
            #    Ibar += sigbar2*self.I[i]+self.sigma2[i]

            #chi2 = 0
            #chi2_v2 = 0
            #for i in range(len(self.I)):
            #    chi2 += (self.I[i]-Ibar)**2/self.sigma2[i]
            #    chi2_v2 += (self.I[i]-Ibar)**2 * self.I[i]**2 / (Ibar_v2**2 * self.sigma2[i])

            #if len(self.I)>1:
            #    chi2perdof = chi2/(len(self.I)-1)
            #    chi2perdof_v2 = chi2/(len(self.I)-1)

            #print ("V1 : %0.5f +/- %0.5f (chi2 = %0.5f)"%(Ibar,sigbar2,chi2))
            #print ("V2 : %0.5f +/- %0.5f (chi2 = %0.5f)"%(Ibar_v2,sigbar2_v2,chi2_v2))
            #print ("cur:  %0.5f +/- %0.5f"%(self.I[-1],self.sigma2[-1])) 
            #print ("\tIntegral = %0.5f +/- %0.5f (chi2/dof = %0.5f)"%(E,0,chi2dof))
            #print (self.analytic_integral)
            #integral = y.mean().item()
            #variance = torch.sqrt((1/(x.shape[0]))*(y-integral).pow(2).sum()).item()
            #print ("\tIntegral = %0.5f +/- %0.5f"%(integral,variance))


            # Draw determinant function #

            #_ , funcdet = self.model(self.grid.float())
            #_, funcinvdet = self.model.backward(self.grid.float())

            #self.visObject.AddContour(self.grid_x1,self.grid_x2,funcdet.reshape(100,100),"Jacobian det over latent $z$")
            #self.visObject.AddContour(self.grid_x1,self.grid_x2,funcinvdet.reshape(100,100),"Inverse Jacobian det over observed $x$")

#            if (integral>0.9):
#                X = torch.from_numpy(np.c_[x,y])
#                _, invdet = self.model.backward(X.float())
#                print (invdet)
#
#                np.save("X.npy",X.data.numpy())
#                np.save("invdet.npy",invdet.data.numpy())
#                sys.exit()

            if self.bins2D is None:
                self.bins2D, x_edges, y_edges  = np.histogram2d(x.data.numpy()[:,0],x.data.numpy()[:,1],bins=20,range=[[0,1],[0,1]])
                self.bins2D = self.bins2D.T
            else:
                newbins, x_edges, y_edges = np.histogram2d(x.data.numpy()[:,0],x.data.numpy()[:,1],bins=20,range=[[0,1],[0,1]])
                self.bins2D += newbins.T

            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            x_centers, y_centers = np.meshgrid(x_centers,y_centers)
                
            # Plot points #
            if epoch%self.save_plt_interval == 0:
                self.visObject.AddPointSet(z,title="Latent space $z$",color='g')
                self.visObject.AddPointSet(x,title="Observed space $x$",color='b')
                #self.visObject.AddPointSet(z_inv,title="Reversed",color='r')
                self.visObject.AddContour(x_centers,y_centers,self.bins2D,"Cumulative points")
            # Create png file #
                self.visObject.MakePlot(epoch)


parser = argparse.ArgumentParser(description="Integration of 2D function")
parser.add_argument('-e','--epochs', action='store', required=True, type=int,
                    help="Number of epochs")
parser.add_argument('-b','--batch_size', action='store', required=True, type=int,
                    help="Batch size")
parser.add_argument('-lr','--lr', action='store', required=True, type=float,
                    help="Learning rate")
parser.add_argument('-N','--hidden_dim', action='store', required=True, type=int,
                    help="Number of neurons per layer in the coupling layers")
parser.add_argument('-nc','--n_coupling_layers', action='store', required=True, type=int,
                    help="Number of coupling layers")
parser.add_argument('-nh','--n_hidden_layers', action='store', required=True, type=int,
                    help="Number of hidden layers in coupling layers")
parser.add_argument('--save_plt_interval', action='store', required=False, type=int, default=5,
                    help="Frequency for plot saving (default : 5)")
parser.add_argument('--batchnorm', action='store_true', required=False, default=False,
                    help="Wether to apply batchnorm")
parser.add_argument('--dirname', action='store', required=True, type=str,
                    help="Directory name for the plots")

args = parser.parse_args()

instance = Integration2D(epochs             = args.epochs,
                         batch_size         = args.batch_size,
                         lr                 = args.lr,
                         hidden_dim         = args.hidden_dim,
                         n_coupling_layers  = args.n_coupling_layers,
                         n_hidden_layers    = args.n_hidden_layers,
                         save_plt_interval  = args.save_plt_interval,
                         batchnorm          = args.batchnorm,
                         plot_dir_name      = args.dirname)

