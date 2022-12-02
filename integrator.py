import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from divergences import Divergence

class Integrator():
    """
    Class implementing a normalizing flow integrator     
    Free inspiration from :
        https://gitlab.com/i-flow/i-flow/, arXiv:2001.05486 (Tensorflow)
    """
    def __init__(self, func, flow, dist, optimizer, scheduler=None, loss_func=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), **kwargs):
        """ Initialize the normalizing flow integrator. 
        Args:
            - func : function to be integrated
            - flow : normalizing flow (transform object) to be trained
            - dist : distribution to sample the inputs from
            - optimizer : optimizer from PyTorch
            - scheduler (optional) : scheduler from PyTorch for LR decay
            - loss_func : name of loss function to be taken from divergences
            - kwargs : additional arguments for the loss
        """
        self.device = device
        self._func = func
        self.global_step = 0
        self.flow = flow.to(device)
        self.dist: torch.distributions.Distribution = dist
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.divergence = Divergence(**kwargs)
        if loss_func not in self.divergence.avail:
            raise RuntimeError("Requested loss function not found in class methods")
        self.loss_func = getattr(self.divergence, loss_func)
        self.z_mapper = {}

    def train_one_step(self, nsamples, lr=None, points=False,integral=False):
        """ Perform one step of integration and improve the sampling.         
        Args:             
            - nsamples(int): Number of samples to be taken in a training step
            - lr(bool): Flag for returning the current learning rate
            - points(bool): Flag for returning the points produced in the space by the flow
            - integral(bool): Flag for returning the integral value or not.
        Returns: dict with values
            - 'loss': Value of the loss function for this step              
            - 'mean'(optional): Estimate of the integral value
            - 'uncertainty'(optional): Integral statistical uncertainty
            - 'z'(optional): latent phase space set of points
            - 'x'(optional): observed phase space set of points
        """
        # Initialize #
        self.flow.train()
        self.optimizer.zero_grad()
        
        # Sample #
        z = self.dist.sample((nsamples,)).to(self.device)
        # log_prob = self.dist.log_prob(z)
            # In practice for uniform dist, log_prob = 0 and absdet is multiplied by 1
            # But in the future we might change sampling dist so good to have

        # Process #
        x, absdet = self.flow(z)
        #absdet *= torch.exp(log_prob) # P_X(x) = PZ(f^-1(x)) |det(df/dx)|^-1
        y = self._func(x)
        y = y + np.finfo(np.float32).eps
        #log_y = torch.log(y)
        #log_absdet = torch.log(absdet)
        #mean = torch.mean(-log_absdet - log_y)
        #var = torch.var(y * absdet)
        mean = torch.mean(y/absdet)
        var = torch.var(y/absdet)
        y = (y/mean).detach()

        # Backprop #
        loss = self.loss_func(y,absdet)
        #loss = mean
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step)
        self.global_step += 1

        # Integral #
        return_dict = {'loss': loss.to('cpu').item()}
        if lr:
            return_dict['lr'] = self.optimizer.param_groups[0]['lr']
        if integral:
            return_dict['mean'] = mean.to('cpu').item()
            return_dict['uncertainty'] = torch.sqrt(var/(nsamples-1.)).to('cpu').item()
        if points:
            return_dict['z'] = z.to('cpu')
            return_dict['x'] = x.to('cpu')
        return return_dict

    def sample(self, nsamples, latent=False,jacobian=True):
        """ 
        Sample from the trained distribution.
        Args:
            - nsamples(int): Number of points to be sampled.
            - latent(bool): if True returns set of points from input distribution, if False from observed (trained) distribution with their 
            - jacobian(bool): return set of points with associated jacobian in a tuple
        Returns:  tf.tensor of size (nsamples, ndim) of sampled points, and jacobian(optional).
        """
        z = self.dist.sample((nsamples,)).to(self.device)
        if latent:
            return z.to('cpu')
        else:
            x, absdet = self.flow(z)
            if jacobian:
                return (x.to('cpu'), absdet.to('cpu'))
            else:
                return x.to('cpu')

    def integrate(self, nsamples):
        """ 
        Integrate the function with trained distribution.

        This method estimates the value of the integral based on
        Monte Carlo importance sampling. It returns a tuple of two
        tf.tensors. The first one is the mean, i.e. the estimate of
        the integral. The second one gives the variance of the integrand.
        To get the variance of the estimated mean, the returned variance
        needs to be divided by (nsamples -1).

        Args:
            nsamples(int): Number of points on which the estimate is based on.

        Returns:
            tuple of 2 floats: mean and variance

        """
        z = self.dist.sample((nsamples,)).to(self.device)
        with torch.no_grad():
            x, absdet = self.flow(z)
            y = self._func(x)
            mean = torch.mean(y/absdet)
            var = torch.var(y/absdet)

        return mean.to('cpu').item(), torch.sqrt(var/(nsamples-1.)).to('cpu').item()
            
    def sample_weights(self, nsamples, yield_samples=False):
        """ 
        Sample from the trained distribution and return their weights.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Optionally, the drawn samples can be returned, too.

        Args:
            - nsamples (int): Number of samples to be drawn.
            - yield_samples (bool): Also return samples if true.

        Returns:
            true/test: tf.tensor of size (nsamples, 1) of sampled weights
            (samples: tf.tensor of size (nsamples, ndims) of sampled points)

        """
        z = self.dist.sample((nsamples,)).to(self.device)
        x, absdet = self.flow(z)
        y = self._func(x)
        if yield_samples:
            return (y/absdet).to('cpu'), x.to('cpu')

        return (y/absdet).to('cpu')

    def sample_with_context(self, context: np.ndarray, jacobian: bool = False):
        """
        Sample from the trained distribution with context for transform network.

        Args:
            - context (np.ndarray): context for transform network
            - jacobian (bool): return set of points with associated jacobian in a tuple

        Returns:
            tf.tensor of size (context.shape[0], ndim) of sampled points, and jacobian(optional).

        """
        z = self.dist.sample((context.shape[0],)).to(self.device)
        list(map(lambda x: self.z_mapper.update({x[1].tobytes(): x[0]}), zip(z, context)))
        with torch.no_grad():
            x, absdet = self.flow(z, context=torch.tensor(context).to(self.device))
        if jacobian:
            return (x.to('cpu'), absdet.to('cpu'))
        else:
            return x.to('cpu')

    def train_with_context(self, context: np.ndarray, batch_size=100, lr=None, points=False, integral=False) -> list:
        # Initialize #
        self.flow.train()


        # Sample #
        # z = self.dist.sample((context.shape[0],)).to(self.device)
        z = torch.stack([self.z_mapper[row.tobytes()] for row in context[:, :-1]])
        # log_prob = self.dist.log_prob(z)
        # In practice for uniform dist, log_prob = 0 and absdet is multiplied by 1
        # But in the future we might change sampling dist so good to have

        # Process #
        context_x = context[:, :-1]
        context_y = context[:, -1]
        train_result = []
        for batch_x, batch_y, batch_z in DataLoader(dataset=TensorDataset(torch.Tensor(context_x),
                                                                          torch.Tensor(context_y),
                                                                          z),
                                           batch_size=batch_size):

            self.optimizer.zero_grad()
            x, absdet = self.flow(batch_z, batch_x.to(self.device))
            # absdet *= torch.exp(log_prob) # P_X(x) = PZ(f^-1(x)) |det(df/dx)|^-1

            # --------------- START TODO compute loss ---------------
            y = batch_y.to(self.device)
            y = y + np.finfo(np.float32).eps

            log_y = torch.log(y)
            log_absdet = torch.log(absdet)
            #mean = torch.mean(y / absdet)
            mean = torch.mean(-log_absdet - log_y)
            var = torch.var(y * absdet)
            #var = torch.var(y / absdet)
            #y = (y / mean).detach()

            # Backprop #
            #loss = self.loss_func(y, absdet)
            loss = mean

            #print("\t" "Loss = %0.8f" % loss)
            # --------------- END TODO compute loss ---------------

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(self.global_step)

            # Integral #
            return_dict = {'loss': loss.to('cpu').item(), 'epoch': self.global_step}
            self.global_step += 1
            if lr:
                return_dict['lr'] = self.optimizer.param_groups[0]['lr']
            if integral:
                return_dict['mean'] = mean.to('cpu').item()
                return_dict['uncertainty'] = torch.sqrt(var / (context.shape[0] - 1.)).to('cpu').item()
            if points:
                return_dict['z'] = batch_z.to('cpu')
                return_dict['x'] = x.to('cpu')
            train_result.append(return_dict)
        return train_result

