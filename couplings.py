import os
import sys
import torch
import torch.nn as nn

import transform
import splines

class CouplingTransform(transform.Transform):
    """
    A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask.
    Free inspiration from :
        https://github.com/bayesiains/nsf, arXiv:1906.04032 (PyTorch)
        https://gitlab.com/i-flow/i-flow/, arXiv:2001.05486 (Tensorflow)
    """

    def __init__(self,mask,transform_net_create_fn,blob=None):
        """
        Constructor.
        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
            transform_net_create_fn : lambda defining a network based on in_features and out_features
                TODO : might want to include options in the transform definition
            blob : int for number of bins to include in the input one-blob encoding
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError('Mask must be a 1-dim tensor.')
        if mask.numel() <= 0:
            raise ValueError('Mask can\'t be empty.')

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))

        assert self.num_identity_features + self.num_transform_features == self.features

        self.blob = bool(blob)
        if self.blob:
            if not isinstance(blob, int):
                raise ValueError('Blob encoding requires a number of bins')             
            self.nbins_in = int(blob)

        if self.blob:
            self.transform_net = transform_net_create_fn(
                self.num_identity_features * self.nbins_in,
                self.num_transform_features * self._transform_dim_multiplier()
            )
        else:
            self.transform_net = transform_net_create_fn(
                self.num_identity_features,
                self.num_transform_features * self._transform_dim_multiplier()
            )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def one_blob(self,xd):
        binning = (0.5/self.nbins_in) + torch.arange(0., 1.,1./self.nbins_in).repeat(xd.numel())
        binning = binning.reshape(-1,self.num_identity_features,self.nbins_in)
        x = xd.unsqueeze(-1)
        res = torch.exp(((-self.nbins_in*self.nbins_in)/2.) * (binning-x)**2)
        return res.reshape(-1,self.num_identity_features*self.nbins_in)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        if self.blob:
            identity_split_blob = self.one_blob(identity_split)
            transform_params = self.transform_net(identity_split_blob, context)
        else:
            transform_params = self.transform_net(identity_split, context)

        transform_split, absdet = self._coupling_transform_forward(
            inputs=transform_split,
            transform_params=transform_params
        )

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, absdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, absdet = self._coupling_transform_inverse(
            inputs=transform_split,
            transform_params=transform_params
        )
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, absdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()

class AffineCouplingTransform(CouplingTransform):
    """An affine coupling layer that scales and shifts part of the variables.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """
    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, :self.num_transform_features, ...]
        scale = torch.exp(nn.Tanh()(unconstrained_scale))
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        outputs = inputs * scale + shift
        absdet = torch.prod(scale,axis=1)
        return outputs, absdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        outputs = (inputs - shift) / scale
        absdet = torch.prod(1/scale,axis=1)
        return outputs, absdet


class AdditiveCouplingTransform(AffineCouplingTransform):
    """An additive coupling layer, i.e. an affine coupling layer without scaling.
    Reference:
    > L. Dinh et al., NICE:  Non-linear  Independent  Components  Estimation,
    > arXiv:1410.8516, 2014.
    """
    def _transform_dim_multiplier(self):
        return 1

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.ones_like(shift,requires_grad=True)
        return scale, shift


class PiecewiseCouplingTransform(CouplingTransform):
    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, absdet = self._piecewise_cdf(inputs, transform_params, inverse)
        return outputs, absdet.prod(1)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()


class PiecewiseLinearCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    def __init__(self,
                 mask,
                 transform_net_create_fn,
                 blob = None,
                 num_bins=10):
        self.num_bins = num_bins

        super().__init__(mask, transform_net_create_fn,blob)

    def _transform_dim_multiplier(self):
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_pdf = transform_params

        return splines.linear_spline(
            inputs=inputs,
            unnormalized_pdf=unnormalized_pdf,
            inverse=inverse
        )

class PiecewiseQuadraticCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    def __init__(self,
                 mask,
                 transform_net_create_fn,
                 blob = None,
                 num_bins=10,
                 min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        super().__init__(mask, transform_net_create_fn, blob)

    def _transform_dim_multiplier(self):
        return self.num_bins * 2 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:]

        if hasattr(self.transform_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        spline_kwargs = {}
        return splines.quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )


class PiecewiseCubicCouplingTransform(PiecewiseCouplingTransform):
    def __init__(self,
                 mask,
                 transform_net_create_fn,
                 blob = None,
                 num_bins=10,
                 min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        super().__init__(mask, transform_net_create_fn, blob)

    def _transform_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = transform_params[..., 2*self.num_bins][..., None]
        unnorm_derivatives_right = transform_params[..., 2*self.num_bins + 1][..., None]

        if hasattr(self.transform_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        spline_kwargs = {}

        return splines.cubic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            **spline_kwargs
        )

