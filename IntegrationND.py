import os
import sys 
import math
import numpy as np
import torch
import torch.nn as nn
import argparse


from visualize import visualize
import functions
from network import *
from integrator import *
from transform import *
from couplings import *

from catch_cuba import CatchCuba

torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IntegrationND:
    def __init__(self,ndims,epochs,batch_size,lr,hidden_dim,n_hidden_layers,funcname,blob,loss_func,piecewise_bins,save_plt_interval,plot_dir_name,cuba):
        self.ndims              = ndims
        self.epochs             = epochs
        self.batch_size         = batch_size
        self.lr                 = lr
        self.hidden_dim         = hidden_dim
        self.n_hidden_layers    = n_hidden_layers
        self.blob               = blob
        self.piecewise_bins     = piecewise_bins  
        self.loss_func          = loss_func
        self.save_plt_interval  = save_plt_interval
        self.plot_dir_name      = plot_dir_name
        self.function           = getattr(functions,funcname)(n=self.ndims)
        self.use_cuba           = cuba

        self._run()

    @staticmethod
    def binary_list(inval, length):
        """ Convert x into a binary list of length l. """
        return np.array([int(i) for i in np.binary_repr(inval, length)])

    def binary_masks(self,ndims):
        """ Create binary masks for to account for symmetries. 
        See arXiv:2001.05486 (section III.A)"""
        # Count max number of masks required #
        n_masks = int(np.ceil(np.log2(ndims)))
        # Binary representation #
        sub_masks = np.transpose(np.array([self.binary_list(i, n_masks) for i in range(ndims)]))[::-1]
        # Interchange 0 <-> 1 in the mask "
        flip_masks = 1-sub_masks 

        # Combine masks
        masks = np.empty((2*n_masks, ndims))
        masks[0::2] = flip_masks
        masks[1::2] = sub_masks

        return masks

    def create_base_transform(self,mask,coupling_name):
        transform_net_create_fn = lambda in_features, out_features: MLP(in_shape            = [in_features],
                                                                        out_shape           = [out_features],
                                                                        hidden_sizes        = [self.hidden_dim]*self.n_hidden_layers,
                                                                        hidden_activation   = nn.ReLU(),
                                                                        output_activation   = None)
        if coupling_name == 'piecewiseLinear':
            return PiecewiseLinearCouplingTransform(mask,transform_net_create_fn,self.blob,self.piecewise_bins)
        elif coupling_name == 'piecewiseQuadratic':
            return PiecewiseQuadraticCouplingTransform(mask,transform_net_create_fn,self.blob,self.piecewise_bins)
        elif coupling_name == 'piecewiseCubic':
            return PiecewiseCubicCouplingTransform(mask,transform_net_create_fn,self.blob,self.piecewise_bins)
        else:
            raise RuntimeError("Could not find coupling with name %s"%coupling_name)

    def create_flow(self,coupling_name):
        return CompositeTransform([self.create_base_transform(mask,coupling_name) for mask in self.masks])

    def _initialize(self):
        # Masks #
        self.masks = self.binary_masks(self.ndims)
        print ("Number of coupling layers : %d"%len(self.masks))

        # Training #
        self.dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]*self.ndims), torch.tensor([1.0]*self.ndims))
        lr_min = 1e-6

        # Visualization plot #
        self.visObject = visualize('Plots/'+self.plot_dir_name)

        #----- Piecewise Linear Model -----#
        piecewiseLinearFlow = self.create_flow('piecewiseLinear')
        piecewiseLinearOptimizer = torch.optim.Adam(piecewiseLinearFlow.parameters(), lr=self.lr)
        piecewiseLinearScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(piecewiseLinearOptimizer, self.epochs, lr_min)
        self.piecewiseLinearIntegrator = Integrator(func        = self.function,
                                                    flow        = piecewiseLinearFlow,
                                                    dist        = self.dist,
                                                    optimizer   = piecewiseLinearOptimizer,
                                                    scheduler   = piecewiseLinearScheduler,
                                                    loss_func   = self.loss_func)
        #----- Piecewise Quadratic Model -----#
        piecewiseQuadraticFlow = self.create_flow('piecewiseQuadratic')
        piecewiseQuadraticOptimizer = torch.optim.Adam(piecewiseQuadraticFlow.parameters(), lr=self.lr)
        piecewiseQuadraticScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(piecewiseQuadraticOptimizer, self.epochs, lr_min)
        self.piecewiseQuadraticIntegrator = Integrator(func        = self.function,
                                                       flow        = piecewiseQuadraticFlow,
                                                       dist        = self.dist,
                                                       optimizer   = piecewiseQuadraticOptimizer,
                                                       scheduler   = piecewiseQuadraticScheduler,
                                                       loss_func   = self.loss_func)
        #----- Piecewise Cubic Model -----#
        piecewiseCubicFlow = self.create_flow('piecewiseCubic')
        piecewiseCubicOptimizer = torch.optim.Adam(piecewiseCubicFlow.parameters(), lr=self.lr)
        piecewiseCubicScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(piecewiseCubicOptimizer, self.epochs, lr_min)
        self.piecewiseCubicIntegrator = Integrator(func        = self.function,
                                                   flow        = piecewiseCubicFlow,
                                                   dist        = self.dist,
                                                   optimizer   = piecewiseCubicOptimizer,
                                                   scheduler   = piecewiseCubicScheduler,
                                                   loss_func   = self.loss_func)

        #----- Cuba -----#
        if self.use_cuba:
            cuba = CatchCuba(func       = self.function.name,
                             ndim       = self.ndims,
                             nstart     = self.batch_size,
                             nincrease  = 0,
                             maxeval    = self.epochs*self.batch_size,
                             epsrel     = 0)
            self.cuba_points = cuba.getPointSets()
            self.cuba_integral = cuba.getIntegralValues()

        #---- Dictionnary -----#
        self.integrator_dict = {'Uniform'               : None,
                                'Piecewise Linear'     : self.piecewiseLinearIntegrator,
                                'Piecewise Quadratic'  : self.piecewiseQuadraticIntegrator,
                                'Piecewise Cubic'      : self.piecewiseCubicIntegrator}
        if self.use_cuba:
            self.integrator_dict['Cuba'] = None

        self.means      = {k:[] for k in self.integrator_dict.keys()}
        self.errors     = {k:[] for k in self.integrator_dict.keys()}
        self.bins2D     = {k:None for k in self.integrator_dict.keys()}
        self.mean_wgt   = {k:0 for k in self.integrator_dict.keys()}
        self.err_wgt    = {k:0 for k in self.integrator_dict.keys()}
        self.loss       = {k:0 for k in self.integrator_dict.keys()}

        # Analytic #
        self.mean_wgt['analytic'] = self.function.integral
        self.err_wgt['analytic'] = self.function.integral_error

    def _run(self):
        print ("="*80)
        print ("New output dir with name %s"%self.plot_dir_name)

        # Make model #
        self._initialize()

        # Loop over epochs #
        print( self.epochs)
        for epoch in range(1,self.epochs+1):
            print ("Epoch %d/%d [%0.2f%%]"%(epoch,self.epochs,epoch/self.epochs*100))
            # loop over models#
            for model, integrator in self.integrator_dict.items():  
                if model == "Cuba":
                    cuba_res = self.cuba_integral[epoch*self.batch_size] # [mean,error,chi2,dof]
                    cuba_pts = self.cuba_points[epoch*self.batch_size]
                    mean_wgt = cuba_res[0]
                    err_wgt = cuba_res[1]
                    x = np.array(cuba_pts)
                    loss = 0.
                    lr = 0.
                else:
                    if model == "Uniform": # Use uniform sampling
                        x = self.dist.sample((self.batch_size,))
                        y = self.function(x)
                        x = x.data.numpy()
                        mean = torch.mean(y).item()
                        error = torch.sqrt(torch.var(y)/(self.batch_size-1.)).item()
                        loss = 0.
                        lr = 0.
                    else: # use NIS
                        # Integrate on one epoch and produce resuts #
                        result_dict = integrator.train_one_step(self.batch_size,lr=True,integral=True,points=True)
                        loss = result_dict['loss']
                        lr = result_dict['lr']
                        mean = result_dict['mean']
                        error = result_dict['uncertainty']
                        z = result_dict['z'].data.numpy()
                        x = result_dict['x'].data.numpy()

                    # Record values #
                    self.means[model].append(mean)
                    self.errors[model].append(error)

                    # Combine all mean and errors 
                    mean_wgt = np.sum(self.means[model]/np.power(self.errors[model],2),axis=-1)
                    err_wgt = np.sum(1./(np.power(self.errors[model],2)), axis=-1)
                    mean_wgt /= err_wgt
                    err_wgt = 1/np.sqrt(err_wgt)

                # Record loss #
                self.loss[model] = loss
                self.loss['analytic'] = 0
                    
            
                # Record and print #
                self.mean_wgt[model] = mean_wgt
                self.err_wgt[model] = err_wgt

                print("\t"+(model+' ').ljust(25,'.')+("Loss = %0.8f"%loss).rjust(20,' ')+("\t(LR = %0.8f)"%lr).ljust(20,' ')+("Integral = %0.8f +/- %0.8f"%(self.mean_wgt[model],self.err_wgt[model])))


            # Curve for all models #
            dict_loss  = {}
            dict_val   = {}
            dict_error = {}
            for model in self.mean_wgt.keys():
                dict_loss['$Loss^{%s}$'%model] = [self.loss[model],0]
                dict_val['$I^{%s}$'%model] = [self.mean_wgt[model],0]
                dict_error['$\sigma_{I}^{%s}$'%model] = [self.err_wgt[model],0]
            self.visObject.AddCurves(x = epoch,x_err = 0, title = "Integral value", dict_val = dict_val)
            self.visObject.AddCurves(x = epoch,x_err = 0, title = "Integral uncertainty", dict_val = dict_error)
            # Plot function output #
            if epoch % self.save_plt_interval == 0:
                self.visObject.MakePlot(epoch)

        # Final printout #
        print ("Models results")
        for model in self.mean_wgt.keys():
            print ('..... '+('Model %s'%model).ljust(40,' ')+'Integral : %0.8f +/- %0.8f'%(self.mean_wgt[model],self.err_wgt[model]))



parser = argparse.ArgumentParser(description="Integration of 2D function")
general = parser.add_argument_group('Global arguments')

general.add_argument('--save_plt_interval', action='store', required=False, type=int, default=5,
                    help="Frequency for plot saving (default : 5)")
general.add_argument('--dirname', action='store', required=True, type=str,
                    help="Directory name for the plots")

NIS = parser.add_argument_group('Arguments for the integration for the Neural Importance Sampling')
NIS.add_argument('-f','--function', action='store', required=True,
                    help="Name of the function in functions.py to use for integration")
NIS.add_argument('-e','--epochs', action='store', required=True, type=int,
                    help="Number of epochs")
NIS.add_argument('-b','--batch_size', action='store', required=True, type=int,
                    help="Batch size")
NIS.add_argument('-lr','--lr', action='store', required=True, type=float,
                    help="Learning rate")
NIS.add_argument('--ndims', action='store', required=True, type=int,
                    help="Integration dimension")
NIS.add_argument('--hidden_dim', action='store', required=True, type=int,
                    help="Number of neurons per layer in the coupling layers")
NIS.add_argument('--n_hidden_layers', action='store', required=True, type=int,
                    help="Number of hidden layers in coupling layers")
NIS.add_argument('--blob', action='store', required=False, type=int,
                    help="Number of bins for blob-encoding (default = None)")
NIS.add_argument('--piecewise', action='store', required=False, type=int, default=10,
                    help="Number of bins for piecewise polynomial coupling (default = 10)")
NIS.add_argument('--loss', action='store', required=False, type=str, default="MSE",
                    help="Name of the loss function in divergences (default = MSE)")
NIS.add_argument('--cuba', action='store_true', required=False, default=False,
                    help="Wether to use integration with Cuba (default = False)")

cuba = parser.add_argument_group('Arguments for the integration using cuba')

args = parser.parse_args()

instance = IntegrationND(epochs             = args.epochs,
                         batch_size         = args.batch_size,
                         lr                 = args.lr,
                         ndims              = args.ndims,
                         hidden_dim         = args.hidden_dim,
                         n_hidden_layers    = args.n_hidden_layers,
                         blob               = args.blob,
                         piecewise_bins     = args.piecewise,
                         loss_func          = args.loss,
                         save_plt_interval  = args.save_plt_interval,
                         plot_dir_name      = args.dirname,
                         funcname           = args.function,
                         cuba               = args.cuba)

