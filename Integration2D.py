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
from utils import pyhocon_wrapper

torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args(arg=sys.argv[1:]):
    train_parser = argparse.ArgumentParser(
        description='Application for model training')

    train_parser.add_argument(
        '-c', '--config', required=True,
        help='Configuration file path')

    return train_parser.parse_args(arg)


class Integration2D:
    def __init__(self, cfg):
        self.config = cfg
        self.epochs = self.config.get_int('train.epochs')
        self.batch_size = self.config.get_int('train.batch_size')
        self.lr = self.config.get_float('train.learning_rate')
        self.hidden_dim = self.config.get_int('train.num_hidden_dims')
        self.n_coupling_layers = self.config.get_int('train.num_coupling_layers')
        self.n_hidden_layers = self.config.get_int('train.num_hidden_layers')
        self.blob = self.config.get_int('train.num_blob_bins', 0)
        self.piecewise_bins = self.config.get_int('train.num_piecewise_bins', 10)
        self.loss_func = self.config.get_string('train.loss', 'MSE')
        self.save_plt_interval = self.config.get_int('logging.save_plt_interval', 5)
        self.plot_dir_name = self.config.get_string('logging.plot_dir_name')
        self.function = getattr(functions, self.config.get_string('train.function'))(n=2)

        self._run()

    def create_base_transform(self,i,coupling_name):
        mask = [i%2,(i+1)%2]
        transform_net_create_fn = lambda in_features, out_features: MLP(in_shape            = [in_features],
                                                                        out_shape           = [out_features],
                                                                        hidden_sizes        = [self.hidden_dim]*self.n_hidden_layers,
                                                                        hidden_activation   = nn.ReLU(),
                                                                        output_activation   = None)
        if coupling_name == 'additive':
            return AdditiveCouplingTransform(mask,transform_net_create_fn,self.blob)
        elif coupling_name == 'affine':
            return AffineCouplingTransform(mask,transform_net_create_fn,self.blob)
        elif coupling_name == 'piecewiseLinear':
            return PiecewiseLinearCouplingTransform(mask,transform_net_create_fn,self.blob,self.piecewise_bins)
        elif coupling_name == 'piecewiseQuadratic':
            return PiecewiseQuadraticCouplingTransform(mask,transform_net_create_fn,self.blob,self.piecewise_bins)
        elif coupling_name == 'piecewiseCubic':
            return PiecewiseCubicCouplingTransform(mask,transform_net_create_fn,self.blob,self.piecewise_bins)
        else:
            raise RuntimeError("Could not find coupling with name %s"%coupling_name)

    def create_flow(self,coupling_name):
        return CompositeTransform([self.create_base_transform(i,coupling_name) for i in range(self.n_coupling_layers)])

    def _initialize(self):
        # Training #
        self.dist = torch.distributions.uniform.Uniform(torch.tensor([0.0,0.0]), torch.tensor([1.0,1.0]))
        lr_min = 1e-6

        # Visualization plot #
        self.visObject = visualize(self.plot_dir_name)

        self.grid_x1 , self.grid_x2 = torch.meshgrid(torch.linspace(0,1,100),torch.linspace(0,1,100))
        grid = torch.cat([self.grid_x1.reshape(-1,1),self.grid_x2.reshape(-1,1)],axis=1)
        self.func_out = self.function(grid).reshape(100,100)

        #----- Additive Model -----#
#        additiveFlow = self.create_flow('additive')
#        additiveOptimizer = torch.optim.Adam(additiveFlow.parameters(), lr=self.lr)
#        additiveScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(additiveOptimizer, self.epochs, lr_min)
#        self.additiveIntegrator = Integrator(func        = self.function,
#                                             flow        = additiveFlow,
#                                             dist        = self.dist,
#                                             optimizer   = additiveOptimizer,
#                                             scheduler   = additiveScheduler,
#                                             loss_func   = self.loss_func)
        #----- Affine Model -----#
        affineFlow = self.create_flow('affine')
        affineOptimizer = torch.optim.Adam(affineFlow.parameters(), lr=self.lr)
        affineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(affineOptimizer, self.epochs, lr_min)
        self.affineIntegrator = Integrator(func        = self.function,
                                           flow        = affineFlow,
                                           dist        = self.dist,
                                           optimizer   = affineOptimizer,
                                           scheduler   = affineScheduler,
                                           loss_func   = self.loss_func)
        #----- Piecewise Linear Model -----#
#        piecewiseLinearFlow = self.create_flow('piecewiseLinear')
#        piecewiseLinearOptimizer = torch.optim.Adam(piecewiseLinearFlow.parameters(), lr=self.lr)
#        piecewiseLinearScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(piecewiseLinearOptimizer, self.epochs, lr_min)
#        self.piecewiseLinearIntegrator = Integrator(func        = self.function,
#                                                    flow        = piecewiseLinearFlow,
#                                                    dist        = self.dist,
#                                                    optimizer   = piecewiseLinearOptimizer,
#                                                    scheduler   = piecewiseLinearScheduler,
#                                                    loss_func   = self.loss_func)
        #----- Piecewise Quadratic Model -----#
#        piecewiseQuadraticFlow = self.create_flow('piecewiseQuadratic')
#        piecewiseQuadraticOptimizer = torch.optim.Adam(piecewiseQuadraticFlow.parameters(), lr=self.lr)
#        piecewiseQuadraticScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(piecewiseQuadraticOptimizer, self.epochs, lr_min)
#        self.piecewiseQuadraticIntegrator = Integrator(func        = self.function,
#                                                       flow        = piecewiseQuadraticFlow,
#                                                       dist        = self.dist,
#                                                       optimizer   = piecewiseQuadraticOptimizer,
#                                                       scheduler   = piecewiseQuadraticScheduler,
#                                                       loss_func   = self.loss_func)
        #----- Piecewise Cubic Model -----#
#        piecewiseCubicFlow = self.create_flow('piecewiseCubic')
#        piecewiseCubicOptimizer = torch.optim.Adam(piecewiseCubicFlow.parameters(), lr=self.lr)
#        piecewiseCubicScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(piecewiseCubicOptimizer, self.epochs, lr_min)
#        self.piecewiseCubicIntegrator = Integrator(func        = self.function,
#                                                   flow        = piecewiseCubicFlow,
#                                                   dist        = self.dist,
#                                                   optimizer   = piecewiseCubicOptimizer,
#                                                   scheduler   = piecewiseCubicScheduler,
#                                                   loss_func   = self.loss_func)

        #---- Dictioniary -----#
        self.integrator_dict = {'Uniform'               : None,
                                #'Additive'             : self.additiveIntegrator,
                                'Affine'               : self.affineIntegrator}
                                #'Piecewise Linear'     : self.piecewiseLinearIntegrator,
                                #'Piecewise Quadratic'  : self.piecewiseQuadraticIntegrator,
                                #'Piecewise Cubic'      : self.piecewiseCubicIntegrator}

        self.means = {k:[] for k in self.integrator_dict.keys()}
        self.errors = {k:[] for k in self.integrator_dict.keys()}
        self.bins2D = {k:None for k in self.integrator_dict.keys()}
        self.mean_wgt = {k:0 for k in self.integrator_dict.keys()}
        self.err_wgt = {k:0 for k in self.integrator_dict.keys()}

        #----- Analytic -----#
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

                # Record and print #
                self.mean_wgt[model] = mean_wgt
                self.err_wgt[model] = err_wgt

                print("\t"+(model+' ').ljust(25,'.')+("Loss = %0.8f"%loss).rjust(20,' ')+("\t(LR = %0.8f)"%lr).ljust(20,' ')+("Integral = %0.8f +/- %0.8f"%(self.mean_wgt[model],self.err_wgt[model])))

                # Visualization #
                self.visObject.AddCurves(x = epoch,x_err = 0, title = model+" model",
                                         dict_val = {'$I_{numeric}$': [mean_wgt,err_wgt],'$I_{analytic}$':[self.function.integral,self.function.integral_error]})
                if self.bins2D[model] is None:
                    self.bins2D[model], x_edges, y_edges  = np.histogram2d(x[:,0],x[:,1],bins=20,range=[[0,1],[0,1]])
                else:
                    newbins, x_edges, y_edges = np.histogram2d(x[:,0],x[:,1],bins=20,range=[[0,1],[0,1]])
                    self.bins2D[model] += newbins.T
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                x_centers, y_centers = np.meshgrid(x_centers,y_centers)

                if epoch % self.save_plt_interval == 0:
                    self.visObject.AddPointSet(x,title="Observed $x$ %s"%model,color='b')
                    self.visObject.AddContour(x_centers,y_centers,self.bins2D[model],"Cumulative %s"%model)

            # Curve for all models #
            dict_val = {}
            for model in self.mean_wgt.keys():
                dict_val['$\sigma_{I}^{%s}$'%model] = [self.err_wgt[model],0]
            self.visObject.AddCurves(x = epoch,x_err = 0, title = "Value error", dict_val = dict_val)
            # Plot function output #
            if epoch % self.save_plt_interval == 0:
                self.visObject.AddPointSet(z,title="Latent space $z$",color='b')
                self.visObject.AddContour(self.grid_x1,self.grid_x2,self.func_out,"Target function : "+self.function.name)
                self.visObject.MakePlot(epoch)

        # Final printout #
        print ("Models results")
        for model in self.mean_wgt.keys():
            print ('..... '+('Model %s'%model).ljust(40,' ')+'Integral : %0.8f +/- %0.8f'%(self.mean_wgt[model],self.err_wgt[model]))


options = parse_args()
config = pyhocon_wrapper.parse_file(options.config)

instance = Integration2D(config)

