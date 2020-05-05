import sys
import numpy as no
import torch


class Integrator():
    def __init__(self, func, flow, dist, optimizer, scheduler=None, loss_func=None, **kwargs):
        """ Initialize the normalizing flow integrator. """         
        self._func          = func         
        self.global_step    = 0         
        self.flow           = flow
        self.dist           = dist
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.loss_func      = loss_func

    def train_one_step(self, nsamples, lr=None,points=False,integral=False):
        # Initialize #
        self.flow.train()
        self.optimizer.zero_grad()
        
        # Sample #
        z = self.dist.sample((nsamples,)) 

        # Process #
        x, absdet = self.flow(z)
        y = self._func(x)
        mean = torch.mean(y/absdet)
        var = torch.var(y/absdet)
        y = y.detach()/mean

        # Backprop #
        loss = self.loss_func(y,absdet)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step)
        self.global_step += 1

        # Integral #
        return_dict = {'loss':loss.item()}
        if lr:
            return_dict['lr'] = self.optimizer.param_groups[0]['lr']
        if integral:
            return_dict['mean'] = mean.item()
            return_dict['error'] = torch.sqrt(var/(nsamples-1.)).item()
        if points:
            return_dict['z'] = z
            return_dict['x'] = x
        return return_dict

    def integrate(self, nsamples):
        z = self.dist.sample((nsamples,))
        with torch.no_grad():
            x, absdet = self.flow(z)
            y = self._func(x)
            mean = torch.mean(y/absdet)
            var = torch.var(y/absdet)

        return mean.item(),torch.sqrt(var/(nsamples-1.)).item()
            
        
        
        
