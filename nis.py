import argparse
import os
import time
import functions
import torch
import numpy as np
import logging

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict
from experiment_config import ExperimentConfig
from utils import pyhocon_wrapper
from visualize import visualize, FunctionVisualizer
from integrator import Integrator
from transform import CompositeTransform
from network import MLP, UNet
from couplings import (
    PiecewiseLinearCouplingTransform,
    PiecewiseQuadraticCouplingTransform,
    PiecewiseCubicCouplingTransform,
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)


class NeuralImportanceSampling:
    def __init__(self, _config: ExperimentConfig):
        """
        _config: ExperimentConfig
        mode: ['server', 'experiment'] - if mode==server, don't use dimension reduction and function in visualization
        """
        self.config = _config
        self.logs_dir = os.path.join("logs", self.config.experiment_dir_name)
        self.integrator = None
        self.visualize_object = None
        self.function_visualizer = None
        self.time = time.time()
        self.num_frame = 0

        # need for gradient accumulation: we apply optimizer.step() only once after the last training call
        self.train_sampling_call_difference = 0
        self.z_buffer = None
        self.context_buffer = None

        # self.visualize_point = VisualizePoint(index=0.5, plot_step=10, nis=self.nis)

    def initialize(self, mode="server"):
        """
        mode: ['server', 'experiment'] - if mode==server, don't use dimension reduction and function in visualization
        """
        self.function: functions.Function = getattr(functions, self.config.funcname)(
            n=self.config.ndims
        )
        # masks = self.create_binary_mask(self.config.ndims)
        masks = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1]]
        if self.config.features_mode == "all_features":
            num_context_features = self.config.num_context_features
        elif self.config.features_mode == "xyz":
            num_context_features = 3
        else:
            num_context_features = 0
        flow = CompositeTransform(
            [
                self.create_base_transform(
                    mask=mask,
                    coupling_name=self.config.coupling_name,
                    hidden_dim=self.config.hidden_dim,
                    n_hidden_layers=self.config.n_hidden_layers,
                    blob=self.config.blob,
                    piecewise_bins=self.config.piecewise_bins,
                    num_context_features=num_context_features,
                    network_type=self.config.network_type,
                )
                for mask in masks
            ]
        )
        coef_net = UNet(
                in_features=6,
                out_features=1,
                max_hidden_features=self.config.hidden_dim,
                num_layers=self.config.n_hidden_layers,
                nonlinearity=nn.ReLU(),
                output_activation=nn.Sigmoid(),
            )
        dist = torch.distributions.uniform.Uniform(
            torch.tensor([0.0] * self.config.ndims),
            torch.tensor([1.0] * self.config.ndims),
        )
        optimizer = torch.optim.Adam(flow.parameters(), lr=self.config.lr)
        self.integrator = Integrator(
            func=self.function,
            flow=flow,
            dist=dist,
            optimizer=optimizer,
            scheduler=None,
            loss_func=self.config.loss_func,
            features_mode=self.config.features_mode,
            coef_net=coef_net,
        )

        self.means = []
        self.errors = []
        self.loss = []
        if mode == "server":
            visualize_function = None
        else:
            visualize_function = self.function
        if self.config.save_plots and self.config.ndims >= 2:
            self.visualize_object = visualize(os.path.join(self.logs_dir, "plots"))
            self.function_visualizer = FunctionVisualizer(
                vis_object=self.visualize_object,
                function=visualize_function,
                input_dimension=self.config.ndims,
                max_plot_dimension=self.config.plot_dimension,
            )
        if self.config.use_tensorboard:
            if self.config.wandb_project is not None:
                import wandb

                wandb.tensorboard.patch(root_logdir=self.logs_dir)
                wandb.init(
                    project=self.config.wandb_project,
                    config=asdict(self.config),
                    sync_tensorboard=True,
                    entity="neural_importance_sampling",
                )
            self.tb_writer = SummaryWriter(log_dir=self.logs_dir)
            self.tb_writer.add_text(
                "Config",
                "\n".join(
                    [f"{k.rjust(20, ' ')}: {v}" for k, v in asdict(self.config).items()]
                ),
            )

    def create_base_transform(
        self,
        mask,
        coupling_name,
        hidden_dim,
        n_hidden_layers,
        blob,
        piecewise_bins,
        num_context_features=0,
        network_type="MLP",
    ):
        if network_type.lower() == "mlp":
            def transform_net_create_fn(in_features, out_features): return MLP(
                in_shape=[in_features],
                out_shape=[out_features],
                hidden_sizes=[hidden_dim] * n_hidden_layers,
                hidden_activation=nn.ReLU(),
                output_activation=None,
            )
        elif network_type.lower() == "unet":
            def transform_net_create_fn(in_features, out_features): return UNet(
                in_features=in_features,
                out_features=out_features,
                max_hidden_features=256,
                num_layers=n_hidden_layers,
                nonlinearity=nn.ReLU(),
                output_activation=None,
            )
        else:
            raise ValueError(
                f"network_type argument should be in [mlp, unet], but given {network_type}"
            )

        if coupling_name == "additive":
            return AdditiveCouplingTransform(
                mask,
                transform_net_create_fn,
                blob,
                num_context_features=num_context_features,
            )
        elif coupling_name == "affine":
            return AffineCouplingTransform(
                mask,
                transform_net_create_fn,
                blob,
                num_context_features=num_context_features,
            )
        elif coupling_name == "piecewiseLinear":
            return PiecewiseLinearCouplingTransform(
                mask,
                transform_net_create_fn,
                blob,
                piecewise_bins,
                num_context_features=num_context_features,
            )
        elif coupling_name == "piecewiseQuadratic":
            return PiecewiseQuadraticCouplingTransform(
                mask,
                transform_net_create_fn,
                blob,
                piecewise_bins,
                num_context_features=num_context_features,
            )
        elif coupling_name == "piecewiseCubic":
            return PiecewiseCubicCouplingTransform(
                mask,
                transform_net_create_fn,
                blob,
                piecewise_bins,
                num_context_features=num_context_features,
            )
        else:
            raise RuntimeError("Could not find coupling with name %s" % coupling_name)

    def create_binary_mask(self, ndims):
        """Create binary masks for to account for symmetries.
        See arXiv:2001.05486 (section III.A)"""
        # Count max number of masks required #
        n_masks = int(np.ceil(np.log2(ndims)))
        # Binary representation #

        def binary_list(inval, length):
            """Convert x into a binary list of length l."""
            return np.array([int(i) for i in np.binary_repr(inval, length)])

        sub_masks = np.transpose(
            np.array([binary_list(i, n_masks) for i in range(ndims)])
        )[::-1]
        # Interchange 0 <-> 1 in the mask "
        flip_masks = 1 - sub_masks

        # Combine masks
        masks = np.empty((2 * n_masks, ndims))
        masks[0::2] = flip_masks
        masks[1::2] = sub_masks

        return masks

    def get_samples(self, context):
        # pdf_light_sample = self.integrator.sample_with_context(context, inverse=True)
        [samples, pdf, coef] = self.integrator.sample_with_context(context, inverse=False)
        pdf_light_sample = torch.ones(pdf.size())
        return [samples, pdf_light_sample, pdf, coef]

    def train(self, context):
        z = torch.stack(self.integrator.generate_z_by_context(context))
        context = torch.tensor(context)
        if self.z_buffer is None:
            self.z_buffer = z
        else:
            self.z_buffer = torch.cat((self.z_buffer, z), 0)
        # print("Z buffer length: " + str(len(self.z_buffer)))
        if self.context_buffer is None:
            self.context_buffer = context
        else:
            self.context_buffer = torch.cat((self.context_buffer, context), 0)

        # if self.train_sampling_call_difference == 1:
        if self.context_buffer.size()[0] > self.config.max_train_buffer_size:
            self.context_buffer = self.context_buffer[
                -self.config.max_train_buffer_size:
            ]
            self.z_buffer = self.z_buffer[-self.config.max_train_buffer_size:]
        for epoch in range(self.config.num_training_steps):
            start = time.time()
            indices = torch.randperm(len(self.context_buffer))[
                : self.config.num_samples_per_training_step
            ]
            epoch_z_context = (self.z_buffer[indices], self.context_buffer[indices])
            logging.info(f"epoch_z_context time: {time.time() - start}")
            start = time.time()
            train_result = self.integrator.train_with_context(
                z_context=epoch_z_context,
                lr=False,
                integral=True,
                points=True,
                batch_size=self.config.batch_size,
                apply_optimizer=not self.config.gradient_accumulation,
            )

            logging.info(f"Epoch time: {time.time() - start}")
            if (
                self.train_sampling_call_difference == 1
            ):  # Test for first bounce (Just check middle pixel)
                for epoch_result in train_result:
                    if self.visualize_object:
                        self.visualize_train_step(epoch_result)
                    if self.config.use_tensorboard:
                        self.log_tensorboard_train_step(epoch_result)

        if self.config.gradient_accumulation:
            self.integrator.apply_optimizer()

        if self.train_sampling_call_difference == 1:
            self.integrator.z_mapper = {}  # Be careful with z_mapper

        # print("Frame computed: ", time.time() - self.time)
        self.time = time.time()

    def visualize_train_step(self, train_result):

        self.means.append(train_result["mean"])
        self.errors.append(train_result["uncertainty"])
        self.loss.append(train_result["loss"])
        mean_wgt = np.sum(self.means / np.power(self.errors, 2), axis=-1)
        err_wgt = np.sum(1.0 / (np.power(self.errors, 2)), axis=-1)
        mean_wgt /= err_wgt
        err_wgt = 1 / np.sqrt(err_wgt)

        # dict_val = {"$I^{%s}$" % self.config.coupling_name: [mean_wgt, 0]}
        # dict_error = {"$\sigma_{I}^{%s}$" % self.config.coupling_name: [err_wgt, 0]}
        # dict_loss = {"$I^{I}^{%s}$": [err_wgt, 0]}
        # self.visualize_object.AddCurves(x=train_result['epoch'], x_err=0, title="Integral value",
        #                                dict_val=dict_val)
        # self.visualize_object.AddCurves(x=train_result['epoch'], x_err=0, title="Integral uncertainty",
        #                                dict_val=dict_error)
        # self.visualize_object.AddCurves(x=train_result['epoch'], x_err=0, title="Loss",
        #                                dict_val=dict_loss)
        if (
            self.config.save_plots and self.config.ndims >= 2
        ):  # if 2D -> visualize distribution
            visualize_x = self.function_visualizer.add_trained_function_plot(
                x=train_result["x"].detach().numpy(),
                plot_name="Cumulative %s" % self.config.coupling_name,
            )
            self.visualize_object.AddPointSet(
                visualize_x,
                title="Observed $x$ %s" % self.config.coupling_name,
                color="b",
            )
            # self.visualize_object.AddPointSet(train_result['z'], title="Latent space $z$", color='b')

            # grid_x1, grid_x2 = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
            # grid = torch.cat([grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1)], axis=1)
            # func_out = self.function(grid).reshape(100, 100)
            # self.visualize_object.AddContour(grid_x1, grid_x2, func_out,
            #                                 "Target function : " + self.function.name)

            # Plot function output #
        if self.num_frame % self.config.save_plt_interval == 0:
            if self.config.save_plots and self.config.ndims >= 2:
                self.visualize_object.AddPointSet(
                    train_result["z"], title="Latent space $z$", color="b"
                )

            self.visualize_object.MakePlot(self.num_frame)

    def log_tensorboard_train_step(self, train_result):
        self.tb_writer.add_scalar(
            "Train/Loss", train_result["loss"], self.integrator.global_step
        )
        self.tb_writer.add_scalar(
            "Train/Integral", train_result["mean_wgt"], self.integrator.global_step
        )
        self.tb_writer.add_scalar(
            "Train/LR", train_result["lr"], self.integrator.global_step
        )

    def run_experiment(self):

        visObject = visualize(os.path.join(self.logs_dir, 'plots'))

        if self.config.use_tensorboard:
            if self.config.wandb_project is not None:
                import wandb
                wandb.tensorboard.patch(root_logdir=logs_dir)
                wandb.init(project=config.wandb_project, config=asdict(config), sync_tensorboard=True,
                           entity="neural_importance_sampling")
            tb_writer = SummaryWriter(log_dir=logs_dir)
            tb_writer.add_text("Config",
                               '\n'.join([f"{k.rjust(20, ' ')}: {v}" for k, v in asdict(self.config).items()]))

        if self.config.ndims == 2:  # if 2D -> prepare x1,x2 gird for visualize
            grid_x1, grid_x2 = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
            grid = torch.cat([grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1)], axis=1)
            func_out = np.flip(self.function(grid).reshape(100, 100).numpy()).T

        means = []
        errors = []
        for epoch in range(1, self.config.epochs + 1):
            print("Epoch %d/%d [%0.2f%%]" % (epoch, self.config.epochs, epoch / self.config.epochs * 100))

            # Integrate on one epoch and produce resuts #
            result_dict = self.integrator.train_one_step(self.config.batch_size, lr=True, integral=True, points=True)
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

            print("\t" + (self.config.coupling_name + ' ').ljust(25, '.') + ("Loss = %0.8f" % loss).rjust(20, ' ') + (
                    "\t(LR = %0.8f)" % lr).ljust(20, ' ') + (
                          "Integral = %0.8f +/- %0.8f" % (mean_wgt, err_wgt)))

            # dict_loss = {'$Loss^{%s}$' % self.config.coupling_name: [loss, 0]}
            dict_val = {'$I^{%s}$' % self.config.coupling_name: [mean_wgt, 0]}
            dict_error = {'$\sigma_{I}^{%s}$' % self.config.coupling_name: [err_wgt, 0]}

            if self.config.use_tensorboard:
                tb_writer.add_scalar('Train/Loss', loss, epoch)
                tb_writer.add_scalar('Train/Integral', mean_wgt, epoch)
                tb_writer.add_scalar('Train/LR', lr, epoch)

            # Plot function output #
            if epoch % self.config.save_plt_interval == 0:
                visObject.AddCurves(x=epoch, x_err=0, title="Integral value", dict_val=dict_val)
                visObject.AddCurves(x=epoch, x_err=0, title="Integral uncertainty", dict_val=dict_error)
                visObject.AddPointSet(x, title="Observed $x$ %s" % self.config.coupling_name, color='b')
                visObject.AddPointSet(z, title="Latent space $z$", color='b')
                x = self.integrator.sample(100000, jacobian=False).data.numpy()
                for _ in range(10):
                    x = np.vstack((x, self.integrator.sample(100000, jacobian=False).data.numpy()))
                bins, x_edges, y_edges = np.histogram2d(1 - x[:, 0], 1 - x[:, 1], bins=100, range=[[0, 1], [0, 1]],
                                                        density=True)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                x_centers, y_centers = np.meshgrid(x_centers, y_centers)
                visObject.AddContour(x_centers, y_centers, bins, "Cumulative %s" % self.config.coupling_name)
                visObject.AddContour(grid_x1, grid_x2, func_out, "Target function : " + self.function.name)
                visObject.MakePlot(epoch)


def parse_args():
    train_parser = argparse.ArgumentParser(description="Application for model training")

    train_parser.add_argument(
        "-c", "--config", required=True, help="Configuration file path"
    )

    return train_parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    config = pyhocon_wrapper.parse_file(options.config)
    experiment_config = ExperimentConfig.init_from_pyhocon(config)

    experiment_config.num_context_features = 0
    nis = NeuralImportanceSampling(experiment_config)
    nis.initialize()
    nis.run_experiment()
