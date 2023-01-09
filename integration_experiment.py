import argparse
import math
import os
import random
import sys
import time
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
from network import MLP, UNet
from transform import CompositeTransform
from utils import pyhocon_wrapper, utils
from visualize import visualize, FunctionVisualizer, VisualizePoint

# Using by server
import socket
from enum import Enum
from collections import namedtuple
import logging
from collections import OrderedDict

Request = namedtuple('Request', 'name length')


class Mode(Enum):
    UNKNOWN = -1
    TRAIN = 0
    INFERENCE = 1

@dataclass
class ExperimentConfig:
    """
    experiment_dir_name: dir in logs folder
    ndims: Integration dimension
    funcname: Name of the function in functions.py to use for integration
    coupling_name: name of the Coupling Layers using in NIS [piecewiseLinear, piecewiseQuadratic, piecewiseCubic]
    num_context_features: : number of context features in transform net
    hidden_dim: Number of neurons per layer in the coupling layers
    n_hidden_layers: Number of hidden layers in coupling layers
    blob: Number of bins for blob-encoding (default = None)
    piecewise_bins: Number of bins for piecewise polynomial coupling (default = 10)
    lr: Learning rate
    epochs: Number of epochs
    loss_func: Name of the loss function in divergences (default = MSE)
    batch_size: Batch size
    save_plots: save plots if ndims >= 2
    plot_dimension: add 2d or 3d plot
    save_plt_interval: Frequency for plot saving (default : 10)
    wandb_project: Name of wandb project in neural_importance_sampling team
    use_tensorboard: Use tensorboard logging
    """

    experiment_dir_name: str = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ndims: int = 3
    funcname: str = "Gaussian"
    coupling_name: str = "piecewiseQuadratic"
    num_context_features: int = 0
    hidden_dim: int = 10
    n_hidden_layers: int = 3
    network_type: str = "MLP"
    blob: Union[int, None] = None
    piecewise_bins: int = 10
    lr: float = 0.01
    epochs: int = 100
    loss_func: str = "MSE"
    batch_size: int = 2000
    gradient_accumulation: bool = True
    hybrid_sampling: bool = False
    num_training_steps: int = 16
    num_samples_per_training_step: int = 10_000
    max_train_buffer_size: int = 2_000_000

    save_plots: blob = True
    plot_dimension: int = 2
    save_plt_interval: int = 10
    wandb_project: Union[str, None] = None
    use_tensorboard: bool = False
    host: str = '127.0.0.1'
    port: int = 65432

    @classmethod
    def init_from_pyhocon(cls, pyhocon_config: pyhocon_wrapper.ConfigTree):
        return ExperimentConfig(epochs=pyhocon_config.get_int('train.epochs'),
                                batch_size=pyhocon_config.get_int('train.batch_size'),
                                gradient_accumulation=pyhocon_config.get_bool('train.gradient_accumulation', True),
                                lr=pyhocon_config.get_float('train.learning_rate'),
                                hidden_dim=pyhocon_config.get_int('train.num_hidden_dims'),
                                ndims=pyhocon_config.get_int('train.num_coupling_layers'),
                                n_hidden_layers=pyhocon_config.get_int('train.num_hidden_layers'),
                                network_type=pyhocon_config.get_string('train.network_type', 'MLP'),
                                blob=pyhocon_config.get_int('train.num_blob_bins', 0),
                                piecewise_bins=pyhocon_config.get_int('train.num_piecewise_bins', 10),
                                loss_func=pyhocon_config.get_string('train.loss', 'MSE'),
                                save_plt_interval=pyhocon_config.get_int('logging.save_plt_interval', 5),
                                experiment_dir_name=pyhocon_config.get_string('logging.plot_dir_name',
                                                                              cls.experiment_dir_name),
                                hybrid_sampling=pyhocon_config.get_bool('train.hybrid_sampling', False),
                                num_training_steps=pyhocon_config.get_int('train.num_training_steps', 16),
                                num_samples_per_training_step=pyhocon_config.get_int('train.num_samples_per_training_step', 10_000),
                                max_train_buffer_size=pyhocon_config.get_int('train.max_train_buffer_size', 2_000_000),

                                funcname=pyhocon_config.get_string('train.function'),
                                coupling_name=pyhocon_config.get_string('train.coupling_name'),
                                num_context_features=pyhocon_config.get_int('train.num_context_features'),
                                wandb_project=pyhocon_config.get_string('logging.tensorboard.wandb_project', None),
                                use_tensorboard=pyhocon_config.get_bool('logging.tensorboard.use_tensorboard', False),
                                save_plots=pyhocon_config.get_bool('logging.save_plots', False),
                                plot_dimension=pyhocon_config.get_int('logging.plot_dimension', 2),
                                host=pyhocon_config.get_string('server.host', '127.0.0.1'),
                                port=pyhocon_config.get_int('server.port', 65432),
                                )


class TrainServer:
    def __init__(self, _config: ExperimentConfig):
        self.config = _config
        self.host = self.config.host
        self.port = self.config.port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.NUM_CONNECTIONS = 5
        self.BUFSIZE = 8192
        self.connection = None
        self.put_infer = Request(b'PUT INFER', 9)
        self.put_train = Request(b'PUT TRAIN', 9)
        self.put_infer_ok = Request(b'PUT INFER OK', 12)
        self.put_train_ok = Request(b'PUT TRAIN OK', 12)
        self.data_ok = Request(b'DATA OK', 7)

        self.length = 0
        self.raw_data = bytearray()
        self.data = bytearray()
        self.mode = Mode

        self.sock.bind((self.host, self.port))
        self.sock.listen(self.NUM_CONNECTIONS)
        self.nis = NeuralImportanceSampling(_config)
        self.hybrid_sampling = self.config.hybrid_sampling

        self.visualize_point = VisualizePoint(index=0.5, plot_step=10, nis=self.nis)

        self.s1 = 0
        self.s2 = 0
        self.pdf = 0
        self.middle_point = None
        self.num_frame = 0
        self.samples_tensor = None

    def connect(self):
        print(f"Waiting for connection by {self.host}")
        self.connection, address = self.sock.accept()
        print(f"Connected by {self.host} successfully")

    def close(self):
        self.connection.close()

    def receive_length(self):
        self.length = int.from_bytes(self.connection.recv(4), 'little')
        #print(str(self.length))

    def receive_raw(self):
        self.raw_data = bytearray()
        bytes_recd = 0
        while bytes_recd < self.length:
            chunk = self.connection.recv(min(self.length - bytes_recd, self.BUFSIZE))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            self.raw_data.extend(chunk)
            bytes_recd = bytes_recd + len(chunk)

    def send(self, data):
        try:
            self.connection.sendall(data)
        except ConnectionError:
            logging.error(f"Client was disconnected suddenly while sending\n")

    def make_infer(self):
        self.nis.train_sampling_call_difference += 1
        #points = np.frombuffer(self.raw_data, dtype=np.float32).reshape((-1, self.config.num_context_features + 2)) #add vec2 light_sample_dir
        points = np.frombuffer(self.raw_data, dtype=np.float32).reshape(
            (-1, 8 + 2))  # add vec2 light_sample_dir
        if self.hybrid_sampling:
            #pdf_light_samples = utils.get_pdf_by_samples_cosine(points[:, 8:])
            #[samples, pdfs] = utils.get_test_samples_cosine(points)  # lights(vec3), pdfs
            pdf_light_samples = utils.get_pdf_by_samples_uniform(points[:, 8:])
            [samples, pdfs] = utils.get_test_samples_uniform(points)  # lights(vec3), pdfs
        else:
        #if self.nis.train_sampling_call_difference == 1:
            #self.middle_point = points[[int(points.shape[0] * 0.5)-1, int(points.shape[0] * 0.5), int(points.shape[0] * 0.5)+1]]
                #print("Size of image: " + str(points.shape[0]) +" X: " + str(self.middle_point[:, 0]) + " Y: " + str(self.middle_point[:, 1])
                #      + " Z: " + str(self.middle_point[:, 2]))
                #[samples_n, pdf_light_samples_n, pdfs_n] = self.nis.get_samples(middle_point)
            [samples, pdf_light_samples, pdfs] = self.nis.get_samples(points)
            self.samples_tensor = samples.clone().numpy()
            samples[:, 0] = samples[:, 0] * 2 * np.pi
            samples[:, 1] = torch.acos(samples[:, 1])
                #print("s1 max = " + str(torch.max(samples[:, 0]).item()) + "s1 min = " + str(torch.min(samples[:, 0]).item()) +
                #      "s2 max = " + str(torch.max(samples[:, 1]).item()) + "s2 min = " + str(torch.min(samples[:, 1]).item()))
            pdfs = (1 / (2 * np.pi)) / pdfs
            #pdf_light_samples = pdf_light_samples / (2 * np.pi)
                #samples[[int(samples.shape[0] * 0.5)-1, int(samples.shape[0] * 0.5), int(samples.shape[0] * 0.5)+1]] = samples_n
                #pdfs[[int(pdfs.shape[0] * 0.5)-1, int(pdfs.shape[0] * 0.5), int(pdfs.shape[0] * 0.5)+1]] = pdfs_n
                #pdf_light_samples[[int(pdf_light_samples.shape[0] * 0.5)-1, int(pdf_light_samples.shape[0] * 0.5), int(pdf_light_samples.shape[0] * 0.5)+1]] = pdf_light_samples_n
            #if self.nis.train_sampling_call_difference == 1:
            #[self.s1, self.s2] = samples[int(samples.shape[0] * 0.5)].numpy()
                #pdf = pdfs[int(pdfs.shape[0] * 0.5)]
            #self.pdf = pdfs[int(pdfs.shape[0] * 0.5)].numpy()
                #self.visualize_point.add_sample_with_pdf_infer([self.s1, self.s2], 1, "infer")
            #self.visualize_point.add_sample_with_pdf_infer([self.s1 / (2 * np.pi), np.cos(self.s2)], 1, "infer")
            #self.visualize_point.plot_pdf(self.middle_point)
            #self.visualize_point.add_point(self.samples_tensor)
            #logging.debug("s1 = %s, s2 = %s, s_last = %s", samples[0, :].numpy(), samples[1, :].numpy(), samples[-1, :].numpy())
            #logging.debug("pdf1 = %s, pdf2 = %s, pdf_last = %s", pdfs[0].numpy(), pdfs[1].numpy(), pdfs[-1].numpy())

        return [samples, pdf_light_samples, pdfs]  # lights, pdfs

    def make_train(self):
        #context = np.frombuffer(self.raw_data, dtype=np.float32).reshape((-1, self.config.num_context_features + 3))
        context = np.frombuffer(self.raw_data, dtype=np.float32).reshape((-1, 8 + 3))
        context = context[~np.isnan(context).any(axis=1), :]
        #mask = (context[:, 0] + context[:, 1] + context[:, 2] > 0)
        #mask_zero = (context[:, 0] + context[:, 1] + context[:, 2] == 0)
        #non_zero_context = context[mask, :]
        #zero_context = context[mask_zero, :]
        #row_id = random.sample(range(0, zero_context.shape[0] - 1), non_zero_context.shape[0])
        #context = np.concatenate((non_zero_context, zero_context[row_id, :]), axis=0)
        #np.random.shuffle(context)
        if self.hybrid_sampling:
            pass
        else:
        #if self.nis.train_sampling_call_difference == 1:
            #idx_1 = np.where((context[:, 3] == self.middle_point[0, 0]) * (context[:, 4] == self.middle_point[0, 1]) *
            #            (context[:, 5] == self.middle_point[0, 2]))
            #idx_2 = np.where((context[:, 3] == self.middle_point[1, 0]) * (context[:, 4] == self.middle_point[1, 1]) *
            #            (context[:, 5] == self.middle_point[1, 2]))
            #idx_3 = np.where((context[:, 3] == self.middle_point[2, 0]) * (context[:, 4] == self.middle_point[2, 1]) *
            #            (context[:, 5] == self.middle_point[2, 2]))

            #if (not np.any(idx_1)) and (not np.any(idx_2)) and (not np.any(idx_3)):
            #    return
                #tmp = torch.tensor([[self.s1 / (2 * np.pi), np.cos(self.s2)]])
                #y = self.nis.function(self.samples_tensor)

            #context = np.concatenate((context[idx_1], context[idx_2], context[idx_3]), axis=0)
                #lum = 0.2126 * context[:, 0] + 0.7152 * context[:, 1] + 0.0722 * context[:, 2]
            lum = 0.3 * context[:, 0] + 0.3 * context[:, 1] + 0.3 * context[:, 2]
                #lum[0] = y[0].item()
                #lum[1] = y[1].item()
                #lum[2] = y[2].item()
            tdata = context[:, [3, 4, 5, 6, 7, 8, 9, 10]]
            tdata = np.concatenate((tdata, lum.reshape([len(lum), 1])), axis=1, dtype=np.float32)
            train_result = self.nis.train(context=tdata)

            #self.visualize_point.add_sample_with_pdf_train([self.s1 / (2 * np.pi), np.cos(self.s2)], pdf, "train")
        self.num_frame +=1
        print("Frame num: " + str(self.num_frame))
        #non_zero = np.count_nonzero(lum)
        #zero = lum.shape[0] - non_zero
        #p = zero * 100 / lum.shape[0]
        #print("Frame: " + str(self.num_frame) + " Bounce: " + str(self.nis.train_sampling_call_difference) + " Non-zero: " + str(non_zero)
        #      + " Zero: " + str(zero) + " (" + str(p) +"%)")
        self.nis.train_sampling_call_difference -= 1
    def process(self):
        try:
            logging.debug('Mode = %s', self.mode.name)
            logging.debug('Len = %s, Data = %s', self.length, np.frombuffer(self.raw_data, dtype=np.float32))
            if self.mode == Mode.TRAIN:
                self.make_train()
                self.connection.send(self.data_ok.name)
            elif self.mode == Mode.INFERENCE:
                [samples, pdf_light_samples, pdfs] = self.make_infer()
                self.connection.send(self.put_infer.name)
                answer = self.connection.recv(self.put_infer_ok.length)
                if answer == self.put_infer_ok.name:
                    raw_data = bytearray()
                    s = samples.cpu().detach().numpy()
                    pls = pdf_light_samples.cpu().detach().numpy()
                    pls[pls<0] = 0
                    s = np.concatenate((s, pls.reshape([len(pls), 1])), axis=1, dtype=np.float32)
                    p = pdfs.cpu().detach().numpy().reshape([-1, 1])
                    raw_data.extend(np.concatenate((s, p), axis=1).tobytes())
                    self.connection.send(len(raw_data).to_bytes(4, 'little'))
                    self.connection.sendall(raw_data)
                    answer = self.connection.recv(self.data_ok.length)
                    if answer == self.data_ok.name:
                        logging.info('Inference data was sent successfully ...\n')
                    else:
                        logging.error('Inference data wasn\'t sent ...\n')
                else:
                    logging.error('Answer is not equal ' + self.put_ok.name)
            else:
                logging.error('Unknown packet type ...')

        except ConnectionError:
            logging.error(f"Connection failed ...")

    def run(self):
        self.nis.initialize(mode='server')
        self.connect()
        try:
            logging.debug('Server started ...')
            while True:
                cmd = self.connection.recv(self.put_infer.length)
                if cmd == self.put_infer.name:
                    self.mode = Mode.INFERENCE
                    self.connection.send(self.put_infer_ok.name)
                    self.receive_length()
                    self.receive_raw()
                    self.process()
                elif cmd == self.put_train.name:
                    self.mode = Mode.TRAIN
                    self.connection.send(self.put_train_ok.name)
                    self.receive_length()
                    self.receive_raw()
                    self.process()

        except ConnectionError:
            logging.error(f"Connection failed ...")
        finally:
            self.close()


class NeuralImportanceSampling:
    def __init__(self, _config: ExperimentConfig):
        """
        _config: ExperimentConfig
        mode: ['server', 'experiment'] - if mode==server, don't use dimension reduction and function in visualization
        """
        self.config = _config
        self.logs_dir = os.path.join('logs', self.config.experiment_dir_name)
        self.integrator = None
        self.visualize_object = None
        self.function_visualizer = None
        self.time = time.time()

        # need for gradient accumulation: we apply optimizer.step() only once after the last training call
        self.train_sampling_call_difference = 0
        self.z_buffer = None
        self.context_buffer = None

    def initialize(self, mode='server'):
        """
        mode: ['server', 'experiment'] - if mode==server, don't use dimension reduction and function in visualization
        """
        self.function: functions.Function = getattr(functions, self.config.funcname)(n=self.config.ndims)
        #masks = self.create_binary_mask(self.config.ndims)
        masks = [[1., 0.], [0., 1.], [1., 0.], [0., 1]]
        flow = CompositeTransform([self.create_base_transform(mask=mask,
                                                              coupling_name=self.config.coupling_name,
                                                              hidden_dim=self.config.hidden_dim,
                                                              n_hidden_layers=self.config.n_hidden_layers,
                                                              blob=self.config.blob,
                                                              piecewise_bins=self.config.piecewise_bins,
                                                              num_context_features=self.config.num_context_features,
                                                              network_type=self.config.network_type)
                                   for mask in masks])
        dist = torch.distributions.uniform.Uniform(torch.tensor([0.0] * self.config.ndims),
                                                   torch.tensor([1.0] * self.config.ndims))
        optimizer = torch.optim.Adam(flow.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs)
        self.integrator = Integrator(func=self.function,
                                        flow=flow,
                                        dist=dist,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loss_func=self.config.loss_func)

        self.means =  []
        self.errors = []
        self.loss = []
        if mode == "server":
            visualize_function = None
        else:
            visualize_function = self.function
        if self.config.save_plots and self.config.ndims >= 2:
            self.visualize_object = visualize(os.path.join(self.logs_dir, 'plots'))
            self.function_visualizer = FunctionVisualizer(vis_object=self.visualize_object, function=visualize_function,
                                                          input_dimension=self.config.ndims,
                                                          max_plot_dimension=self.config.plot_dimension)
        if self.config.use_tensorboard:
            if self.config.wandb_project is not None:
                import wandb
                wandb.tensorboard.patch(root_logdir=self.logs_dir)
                wandb.init(project=self.config.wandb_project, config=asdict(self.config), sync_tensorboard=True,
                           entity="neural_importance_sampling")
            self.tb_writer = SummaryWriter(log_dir=self.logs_dir)
            self.tb_writer.add_text("Config", '\n'.join([f"{k.rjust(20, ' ')}: {v}" for k, v in asdict(self.config).items()]))

    def create_base_transform(self, mask, coupling_name, hidden_dim, n_hidden_layers, blob, piecewise_bins,
                              num_context_features=0, network_type='MLP'):
        if network_type.lower() == "mlp":
            transform_net_create_fn = lambda in_features, out_features: MLP(in_shape=[in_features],
                                                                            out_shape=[out_features],
                                                                            hidden_sizes=[hidden_dim] * n_hidden_layers,
                                                                            hidden_activation=nn.ReLU(),
                                                                            output_activation=None)
        elif network_type.lower() == 'unet':
            transform_net_create_fn = lambda in_features, out_features: UNet(in_features=in_features,
                                                                             out_features=out_features,
                                                                             max_hidden_features=256,
                                                                             num_layers=n_hidden_layers,
                                                                             nonlinearity=nn.ReLU(),
                                                                             output_activation=nn.Softmax())
        else:
            raise ValueError(f"network_type argument should be in [mlp, unet], but given {network_type}")

        if coupling_name == 'additive':
            return AdditiveCouplingTransform(mask, transform_net_create_fn, blob,
                                             num_context_features=num_context_features)
        elif coupling_name == 'affine':
            return AffineCouplingTransform(mask, transform_net_create_fn, blob,
                                           num_context_features=num_context_features)
        elif coupling_name == 'piecewiseLinear':
            return PiecewiseLinearCouplingTransform(mask, transform_net_create_fn, blob, piecewise_bins,
                                                    num_context_features=num_context_features)
        elif coupling_name == 'piecewiseQuadratic':
            return PiecewiseQuadraticCouplingTransform(mask, transform_net_create_fn, blob, piecewise_bins,
                                                       num_context_features=num_context_features)
        elif coupling_name == 'piecewiseCubic':
            return PiecewiseCubicCouplingTransform(mask, transform_net_create_fn, blob, piecewise_bins,
                                                   num_context_features=num_context_features)
        else:
            raise RuntimeError("Could not find coupling with name %s" % coupling_name)

    def create_binary_mask(self, ndims):
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

    def get_samples(self, context):
        pdf_light_sample = self.integrator.sample_with_context(context, inverse=True)
        [samples, pdf] = self.integrator.sample_with_context(context, inverse=False)
        #pdf_light_sample = torch.ones(pdf.size())
        return [samples, pdf_light_sample, pdf]

    def train(self, context):
        z = torch.stack(self.integrator.generate_z_by_context(context))
        context = torch.tensor(context)
        if self.z_buffer is None:
            self.z_buffer = z
        else:
            self.z_buffer = torch.cat((self.z_buffer, z), 0)
        #print("Z buffer length: " + str(len(self.z_buffer)))
        if self.context_buffer is None:
            self.context_buffer = context
        else:
            self.context_buffer = torch.cat((self.context_buffer, context), 0)

        #if self.train_sampling_call_difference == 1:
        if self.context_buffer.size()[0] > self.config.max_train_buffer_size:
            self.context_buffer = self.context_buffer[-self.config.max_train_buffer_size:]
            self.z_buffer = self.z_buffer[-self.config.max_train_buffer_size:]
        train_results = []
        for epoch in range(self.config.num_training_steps):
            start = time.time()
            indices = torch.randperm(len(self.context_buffer))[:self.config.num_samples_per_training_step]
            epoch_z_context = (self.z_buffer[indices], self.context_buffer[indices])
            logging.info(f"epoch_z_context time: {time.time() - start}")
            start = time.time()
            train_result = self.integrator.train_with_context(z_context=epoch_z_context, lr=False, integral=True,
                                                                  points=True,
                                                                  batch_size=self.config.batch_size,
                                                                  apply_optimizer=not self.config.gradient_accumulation)


            logging.info(f"Epoch time: {time.time() - start}")
            train_results.extend(train_result)
            if self.train_sampling_call_difference == 1:    # Test for first bounce (Just check middle pixel)
                for epoch_result in train_result:
                    if self.visualize_object:
                        self.visualize_train_step(epoch_result)
                    if self.config.use_tensorboard:
                        self.log_tensorboard_train_step(epoch_result)

        if self.config.gradient_accumulation:
            self.integrator.apply_optimizer()

        if self.train_sampling_call_difference == 1:
            self.integrator.z_mapper = {}
        #print("Frame computed: ", time.time() - self.time)
        self.time = time.time()
        return train_results

    def visualize_train_step(self, train_result):

        self.means.append(train_result['mean'])
        self.errors.append(train_result['uncertainty'])
        self.loss.append(train_result['loss'])
        mean_wgt = np.sum(self.means / np.power(self.errors, 2), axis=-1)
        err_wgt = np.sum(1. / (np.power(self.errors, 2)), axis=-1)
        mean_wgt /= err_wgt
        err_wgt = 1 / np.sqrt(err_wgt)

        dict_val = {'$I^{%s}$' % self.config.coupling_name: [mean_wgt, 0]}
        dict_error = {'$\sigma_{I}^{%s}$' % self.config.coupling_name: [err_wgt, 0]}
        dict_loss = {'$I^{I}^{%s}$': [err_wgt, 0]}
        self.visualize_object.AddCurves(x=train_result['epoch'], x_err=0, title="Integral value",
                                        dict_val=dict_val)
        self.visualize_object.AddCurves(x=train_result['epoch'], x_err=0, title="Integral uncertainty",
                                        dict_val=dict_error)
        self.visualize_object.AddCurves(x=train_result['epoch'], x_err=0, title="Loss",
                                        dict_val=dict_error)
        if self.config.save_plots and self.config.ndims >= 2:  # if 2D -> visualize distribution
            visualize_x = self.function_visualizer.add_trained_function_plot(x=train_result['x'].detach().numpy(),
                                                                        plot_name="Cumulative %s" % self.config.coupling_name)
            self.visualize_object.AddPointSet(visualize_x, title="Observed $x$ %s" % self.config.coupling_name, color='b')
            self.visualize_object.AddPointSet(train_result['z'], title="Latent space $z$", color='b')
            # Plot function output #
        if train_result['epoch'] % self.config.save_plt_interval == 1:  #Don't forget fix that. This counter not depends on Epoch, just Frame counter
            if self.config.save_plots and self.config.ndims >= 2:
                self.visualize_object.AddPointSet(train_result['z'], title="Latent space $z$", color='b')

            self.visualize_object.MakePlot(train_result['epoch'])

    def log_tensorboard_train_step(self, train_result):
        self.tb_writer.add_scalar('Train/Loss', train_result['loss'], self.integrator.global_step)
        self.tb_writer.add_scalar('Train/Integral', train_result['mean_wgt'], self.integrator.global_step)
        self.tb_writer.add_scalar('Train/LR', train_result['lr'], self.integrator.global_step)

    def run_experiment(self):

        visObject = visualize(os.path.join(self.logs_dir, 'plots'))

        if self.config.use_tensorboard:
            if self.config.wandb_project is not None:
                import wandb
                wandb.tensorboard.patch(root_logdir=logs_dir)
                wandb.init(project=config.wandb_project, config=asdict(config), sync_tensorboard=True,
                           entity="neural_importance_sampling")
            tb_writer = SummaryWriter(log_dir=logs_dir)
            tb_writer.add_text("Config", '\n'.join([f"{k.rjust(20, ' ')}: {v}" for k, v in asdict(self.config).items()]))

        if self.config.ndims == 2:  # if 2D -> prepare x1,x2 gird for visualize
            grid_x1, grid_x2 = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
            grid = torch.cat([grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1)], axis=1)
            func_out = self.function(grid).reshape(100, 100)

        bins = None
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

            visObject.AddCurves(x=epoch, x_err=0, title="Integral value", dict_val=dict_val)
            visObject.AddCurves(x=epoch, x_err=0, title="Integral uncertainty", dict_val=dict_error)

            if self.config.ndims == 2:  # if 2D -> visualize distribution
                if bins is None:
                    bins, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=20, range=[[0, 1], [0, 1]])
                else:
                    newbins, x_edges, y_edges = np.histogram2d(x[:, 0], x[:, 1], bins=20, range=[[0, 1], [0, 1]])
                    bins += newbins.T
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                x_centers, y_centers = np.meshgrid(x_centers, y_centers)
                visObject.AddPointSet(x, title="Observed $x$ %s" % self.config.coupling_name, color='b')
                visObject.AddContour(x_centers, y_centers, bins, "Cumulative %s" % self.config.coupling_name)
                visObject.AddPointSet(z, title="Latent space $z$", color='b')

            if self.config.use_tensorboard:
                tb_writer.add_scalar('Train/Loss', loss, epoch)
                tb_writer.add_scalar('Train/Integral', mean_wgt, epoch)
                tb_writer.add_scalar('Train/LR', lr, epoch)

            # Plot function output #
            if epoch % self.config.save_plt_interval == 0:
                if self.config.ndims == 2:
                    visObject.AddPointSet(z, title="Latent space $z$", color='b')
                    visObject.AddContour(grid_x1, grid_x2, func_out,
                                         "Target function : " + self.function.name)
                visObject.MakePlot(epoch)


def parse_args(arg=sys.argv[1:]):
    train_parser = argparse.ArgumentParser(
        description='Application for model training')

    train_parser.add_argument(
        '-c', '--config', required=True,
        help='Configuration file path')

    return train_parser.parse_args(arg)


def server_processing(experiment_config):
    server = TrainServer(experiment_config)
    server.run()


if __name__ == '__main__':
    options = parse_args()
    config = pyhocon_wrapper.parse_file(options.config)
    experiment_config = ExperimentConfig.init_from_pyhocon(config)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    server_processing(experiment_config)

    #experiment_config.num_context_features = 0
    #nis = NeuralImportanceSampling(experiment_config)
    #nis.initialize()
    #nis.run_experiment()
