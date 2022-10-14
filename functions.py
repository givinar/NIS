import os
import sys
import torch
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt 

alpha = 0.2
p1 = 0.4
p2 = 0.6
r = 0.25
w = 1/0.004
a = 3
R1 = 0.20
R2 = 0.45

class Gaussian:
    def __init__(self,n,alpha=alpha):
        self.alpha = alpha
        self.n = n
    def __call__(self,x):
        assert self.n == x.shape[1]
        pre = 1./(self.alpha*math.sqrt(math.pi))**self.n
        mu = torch.tensor([0.5]*self.n)
        exp = torch.exp(-(x-mu).pow(2).sum(1)/self.alpha**2)
        return exp*pre
    @property
    def integral(self):
        return math.erf(1/(2*self.alpha))**self.n 
    @property
    def integral_error(self):
        return 0
    @property
    def name(self):
        return "Gaussian"

class Camel:
    def __init__(self,n,alpha=alpha):
        self.alpha = alpha
        self.n = n
    def __call__(self,x):
        assert self.n == x.shape[1]
        pre = 0.5/(self.alpha*math.sqrt(math.pi))**self.n
        mu1 = torch.tensor([1/3]*self.n)
        mu2 = torch.tensor([2/3]*self.n)
        exp1 = torch.exp(-(x-mu1).pow(2).sum(1)/self.alpha**2)
        exp2 = torch.exp(-(x-mu2).pow(2).sum(1)/self.alpha**2)
        return (exp1+exp2)*pre
    @property
    def integral(self):
        return (0.5*(math.erf(1/(3*self.alpha))+math.erf(2/(3*self.alpha))))**self.n 
    @property
    def integral_error(self):
        return 0
    @property
    def name(self):
        return "Camel"

class Circles:
    def __init__(self,n):
        self.n = n
    def __call__(self,x):
        assert(x.shape[1] == 2)
        exp1 = torch.exp(-w*torch.abs((x[...,1]-p2).pow(2)+(x[...,0]-p1).pow(2)-r**2))
        exp2 = torch.exp(-w*torch.abs((x[...,1]-1+p2).pow(2)+(x[...,0]-1+p1).pow(2)-r**2))
        return x[...,1].pow(a)*exp1+(1-x[...,1]).pow(a)*exp2
    @property
    def integral(self):
        return 0.0136848
    @property
    def integral_error(self):
        return 5e-9
    @property
    def name(self):
        return "Circles"

class Ring:
    def __init__(self,n,R1=R1,R2=R2):
        self.n = n
    def __call__(self,x):
        assert(x.shape[1] == 2)
        assert R2>R1
        radius = torch.sum((x-torch.tensor([0.5,0.5])).pow(2),axis=-1)
        out_of_bounds = (radius < R1**2) | (radius > R2**2)
        return torch.where(out_of_bounds,torch.zeros_like(radius),torch.ones_like(radius))
    @property
    def integral(self):
        return math.pi*(R2**2-R1**2)
    @property
    def integral_error(self):
        return 0
    @property
    def name(self):
        return "Ring"

