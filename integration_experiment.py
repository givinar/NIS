import argparse
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from couplings import PiecewiseLinearCouplingTransform, PiecewiseQuadraticCouplingTransform, \
    PiecewiseCubicCouplingTransform, AdditiveCouplingTransform, AffineCouplingTransform
from integrator import Integrator
import functions
from network import MLP
from transform import CompositeTransform
from utils import pyhocon_wrapper
from visualize import visualize, FunctionVisualizer


@dataclass
class ExperimentConfig:
    """
    experiment_dir_name: dir in logs folder
    ndims: Integration dimension
    funcname: Name of the function in functions.py to use for integration
    coupling_name: name of the Coupling Layers using in NIS [piecewiseLinear, piecewiseQuadratic, piecewiseCubic]
    hidden_dim: Number of neurons per layer in the coupling layers
    n_hidden_layers: Number of hidden layers in coupling layers
    blob: Number of bins for blob-encoding (default = None)
    piecewise_bins: Number of bins for piecewise polynomial coupling (default = 10)
    lr: Learning rate
    epochs: Number of epochs
    loss_func: Name of the loss function in divergences (default = MSE)
    batch_size: Batch size
    save_plt_interval: Frequency for plot saving (default : 10)
    wandb_project: Name of wandb project in neural_importance_sampling team
    use_tensorboard: Use tensorboard logging
    save_plots: save plots if ndims >= 2
    plot_dimension: add 2d or 3d plot
    """

    experiment_dir_name: str = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ndims: int = 3
    funcname: str = "Gaussian"
    coupling_name: str = "piecewiseQuadratic"
    hidden_dim: int = 10
    n_hidden_layers: int = 3
    blob: Union[int, None] = None
    piecewise_bins: int = 10
    lr: float = 0.01
    epochs: int = 100
    loss_func: str = "MSE"
    batch_size: int = 2000

    save_plt_interval: int = 10
    wandb_project: Union[str, None] = None
    use_tensorboard: bool = False
    save_plots: blob = True
    plot_dimension: int = 2

    @classmethod
    def init_from_pyhocon(cls, pyhocon_config: pyhocon_wrapper.ConfigTree):
        return ExperimentConfig(epochs=pyhocon_config.get_int('train.epochs'),
                                batch_size=pyhocon_config.get_int('train.batch_size'),
                                lr=pyhocon_config.get_float('train.learning_rate'),
                                hidden_dim=pyhocon_config.get_int('train.num_hidden_dims'),
                                ndims=pyhocon_config.get_int('train.num_coupling_layers'),
                                n_hidden_layers=pyhocon_config.get_int('train.num_hidden_layers'),
                                blob=pyhocon_config.get_int('train.num_blob_bins', 0),
                                piecewise_bins=pyhocon_config.get_int('train.num_piecewise_bins', 10),
                                loss_func=pyhocon_config.get_string('train.loss', 'MSE'),
                                save_plt_interval=pyhocon_config.get_int('logging.save_plt_interval', 5),
                                experiment_dir_name=pyhocon_config.get_string('logging.plot_dir_name', cls.experiment_dir_name),
                                funcname=pyhocon_config.get_string('train.function'),
                                coupling_name=pyhocon_config.get_string('train.coupling_name'),
                                wandb_project=pyhocon_config.get_string('logging.tensorboard.wandb_project', None),
                                use_tensorboard=pyhocon_config.get_bool('logging.tensorboard.use_tensorboard', False),
                                save_plots=pyhocon_config.get_bool('logging.save_plots', False),
                                plot_dimension=pyhocon_config.get_int('logging.plot_dimension', 2),
                                )


def create_base_transform(mask, coupling_name, hidden_dim, n_hidden_layers, blob, piecewise_bins):
    transform_net_create_fn = lambda in_features, out_features: MLP(in_shape=[in_features],
                                                                    out_shape=[out_features],
                                                                    hidden_sizes=[hidden_dim] * n_hidden_layers,
                                                                    hidden_activation=nn.ReLU(),
                                                                    output_activation=None)
    if coupling_name == 'additive':
        return AdditiveCouplingTransform(mask, transform_net_create_fn, blob)
    elif coupling_name == 'affine':
        return AffineCouplingTransform(mask, transform_net_create_fn, blob)
    elif coupling_name == 'piecewiseLinear':
        return PiecewiseLinearCouplingTransform(mask, transform_net_create_fn, blob, piecewise_bins)
    elif coupling_name == 'piecewiseQuadratic':
        return PiecewiseQuadraticCouplingTransform(mask, transform_net_create_fn, blob, piecewise_bins)
    elif coupling_name == 'piecewiseCubic':
        return PiecewiseCubicCouplingTransform(mask, transform_net_create_fn, blob, piecewise_bins)
    else:
        raise RuntimeError("Could not find coupling with name %s" % coupling_name)


def create_binary_mask(ndims):
    """ Create binary masks for to account for symmetries.
            See arXiv:2001.05486 (section III.A)"""
    # Count max number of masks required #
    n_masks = int(np.ceil(np.log2(ndims)))
    # Binary representation #

    def binary_list(inval, length):
        """ Convert x into a binary list of length l. """
        return np.array([int(i) for i in np.binary_repr(inval, length)])
    sub_masks = np.transpose(np.array([binary_list(i, n_masks) for i in range(ndims)]))[::-1]
    # Interchange 0 <-> 1 in the mask "
    flip_masks = 1 - sub_masks

    # Combine masks
    masks = np.empty((2 * n_masks, ndims))
    masks[0::2] = flip_masks
    masks[1::2] = sub_masks

    return masks


def run_experiment(config: ExperimentConfig):

    logs_dir = os.path.join('logs', config.experiment_dir_name)
    visObject = visualize(os.path.join(logs_dir, 'plots'))

    if config.use_tensorboard:
        if config.wandb_project is not None:
            import wandb
            wandb.tensorboard.patch(root_logdir=logs_dir)
            wandb.init(project=config.wandb_project, config=asdict(config), sync_tensorboard=True,
                       entity="neural_importance_sampling")
        tb_writer = SummaryWriter(log_dir=logs_dir)
        tb_writer.add_text("Config", '\n'.join([f"{k.rjust(20, ' ')}: {v}" for k, v in asdict(config).items()]))

    function: functions.Function = getattr(functions, config.funcname)(n=config.ndims)
    masks = create_binary_mask(config.ndims)
    flow = CompositeTransform([create_base_transform(mask=mask,
                                                     coupling_name=config.coupling_name,
                                                     hidden_dim=config.hidden_dim,
                                                     n_hidden_layers=config.n_hidden_layers,
                                                     blob=config.blob,
                                                     piecewise_bins=config.piecewise_bins)
                               for mask in masks])
    dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]*config.ndims),
                                               torch.tensor([1.0]*config.ndims))
    optimizer = torch.optim.Adam(flow.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    integrator = Integrator(func=function,
                            flow=flow,
                            dist=dist,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            loss_func=config.loss_func)

    if config.save_plots and config.ndims >= 2:
        function_visualizer = FunctionVisualizer(vis_object=visObject, function=function, input_dimension=config.ndims,
                                                 max_plot_dimension=config.plot_dimension)

    means = []
    errors = []
    for epoch in range(1, config.epochs + 1):
        print("Epoch %d/%d [%0.2f%%]" % (epoch, config.epochs, epoch / config.epochs * 100))

        # Integrate on one epoch and produce resuts #
        result_dict = integrator.train_one_step(config.batch_size, lr=True, integral=True, points=True)
        loss = result_dict['loss']
        lr = result_dict['lr']
        mean = result_dict['mean']
        error = result_dict['uncertainty']
        z = result_dict['z'].data.numpy()
        x = result_dict['x'].data.numpy()

        # Record values #
        means.append(mean)
        errors.append(error)

        # Combine all mean and errors
        mean_wgt = np.sum(means / np.power(errors, 2), axis=-1)
        err_wgt = np.sum(1. / (np.power(errors, 2)), axis=-1)
        mean_wgt /= err_wgt
        err_wgt = 1 / np.sqrt(err_wgt)

        print("\t" + (config.coupling_name + ' ').ljust(25, '.') + ("Loss = %0.8f" % loss).rjust(20, ' ') + (
                    "\t(LR = %0.8f)" % lr).ljust(20, ' ') + (
                          "Integral = %0.8f +/- %0.8f" % (mean_wgt, err_wgt)))

        # dict_loss = {'$Loss^{%s}$' % config.coupling_name: [loss, 0]}
        dict_val = {'$I^{%s}$' % config.coupling_name: [mean_wgt, 0]}
        dict_error = {'$\sigma_{I}^{%s}$' % config.coupling_name: [err_wgt, 0]}

        visObject.AddCurves(x=epoch, x_err=0, title="Integral value", dict_val=dict_val)
        visObject.AddCurves(x=epoch, x_err=0, title="Integral uncertainty", dict_val=dict_error)

        if config.save_plots and config.ndims >= 2:  # if 2D -> visualize distribution
            visualize_x = function_visualizer.add_trained_function_plot(x=x, plot_name="Cumulative %s" % config.coupling_name)
            visObject.AddPointSet(visualize_x, title="Observed $x$ %s" % config.coupling_name, color='b')
            visObject.AddPointSet(z, title="Latent space $z$", color='b')

        if config.use_tensorboard:
            tb_writer.add_scalar('Train/Loss', loss, epoch)
            tb_writer.add_scalar('Train/Integral', mean_wgt, epoch)
            tb_writer.add_scalar('Train/LR', lr, epoch)

        # Plot function output #
        if epoch % config.save_plt_interval == 0:
            if config.save_plots and config.ndims >= 2:
                visObject.AddPointSet(z, title="Latent space $z$", color='b')
                function_visualizer.add_target_function_plot()
            visObject.MakePlot(epoch)


def parse_args(arg=sys.argv[1:]):
    train_parser = argparse.ArgumentParser(
        description='Application for model training')

    train_parser.add_argument(
        '-c', '--config', required=True,
        help='Configuration file path')

    return train_parser.parse_args(arg)


if __name__ == '__main__':
    options = parse_args()
    config = pyhocon_wrapper.parse_file(options.config)
    experiment_config = ExperimentConfig.init_from_pyhocon(config)
    run_experiment(experiment_config)
