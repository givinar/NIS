import sys

import torch
import numpy as np

from torch.nn import functional as F
from torch import nn

class MLP(nn.Module):
    """A standard multi-layer perceptron."""

    def __init__(self,
                 in_shape,
                 out_shape,
                 hidden_sizes,
                 hidden_activation=torch.relu,
                 output_activation=None):
        """
        Args:
            in_shape: tuple, list or torch.Size, the shape of the input.
            out_shape: tuple, list or torch.Size, the shape of the output.
            hidden_sizes: iterable of ints, the hidden-layer sizes.
            hidden_activation: callable, the activation function.
            activate_output: bool, whether to apply the activation to the output.
        """
        super().__init__()
        self._in_shape = torch.Size(in_shape)
        self._out_shape = torch.Size(out_shape)
        self._hidden_sizes = hidden_sizes
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation 

        if len(hidden_sizes) == 0:
            raise ValueError('List of hidden sizes can\'t be empty.')

        self._input_layer = nn.Linear(np.prod(in_shape), hidden_sizes[0])
        self._hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])
        ])
        self._output_layer = nn.Linear(hidden_sizes[-1], np.prod(out_shape))

    def forward(self, inputs, context=None):
        if inputs.shape[1:] != self._in_shape:
            raise ValueError('Expected inputs of shape {}, got {}.'.format(
                self._in_shape, inputs.shape[1:]))

        inputs = inputs.reshape(-1, np.prod(self._in_shape))
        outputs = self._input_layer(inputs)
        outputs = self._hidden_activation(outputs)

        for hidden_layer in self._hidden_layers:
            outputs = hidden_layer(outputs)
            outputs = self._hidden_activation(outputs)

        outputs = self._output_layer(outputs)
        if self._output_activation:
            outputs = self._output_activation(outputs)
        outputs = outputs.reshape(-1, *self._out_shape)

        return outputs


class UNet(nn.Module):
    def __init__(self,
                 in_features,
                 max_hidden_features,
                 num_layers,
                 out_features,
                 nonlinearity=F.relu):
        super().__init__()

        assert utils.is_power_of_two(max_hidden_features), \
            '\'max_hidden_features\' must be a power of two.'
        assert max_hidden_features // 2 ** num_layers > 1, \
            '\'num_layers\' must be {} or fewer'.format(int(np.log2(max_hidden_features) - 1))

        self.nonlinearity = nonlinearity
        self.num_layers = num_layers

        self.initial_layer = nn.Linear(in_features, max_hidden_features)

        self.down_layers = nn.ModuleList([
            nn.Linear(
                in_features=max_hidden_features // 2 ** i,
                out_features=max_hidden_features // 2 ** (i + 1)
            )
            for i in range(num_layers)
        ])

        self.middle_layer = nn.Linear(
            in_features=max_hidden_features // 2 ** num_layers,
            out_features=max_hidden_features // 2 ** num_layers)

        self.up_layers = nn.ModuleList([
            nn.Linear(
                in_features=max_hidden_features // 2 ** (i + 1),
                out_features=max_hidden_features // 2 ** i
            )
            for i in range(num_layers - 1, -1, -1)
        ])

        self.final_layer = nn.Linear(max_hidden_features, out_features)

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        temps = self.nonlinearity(temps)

        down_temps = []
        for layer in self.down_layers:
            temps = layer(temps)
            temps = self.nonlinearity(temps)
            down_temps.append(temps)

        temps = self.middle_layer(temps)
        temps = self.nonlinearity(temps)

        for i, layer in enumerate(self.up_layers):
            temps += down_temps[self.num_layers - i - 1]
            temps = self.nonlinearity(temps)
            temps = layer(temps)

        return self.final_layer(temps)


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self,
                 features,
                 context_features,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList([
            nn.Linear(features, features)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(
                torch.cat(
                    (temps, self.context_layer(context)),
                    dim=1
                ),
                dim=1
            )
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 context_features=None,
                 num_blocks=2,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(in_features + context_features, hidden_features)
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=hidden_features,
                context_features=context_features,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(
                torch.cat((inputs, context), dim=1)
            )
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs

