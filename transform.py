import sys

import torch
from torch import nn

"""
Free inspiration from :
    https://github.com/bayesiains/nsf, arXiv:1906.04032 (PyTorch)
    https://gitlab.com/i-flow/i-flow/, arXiv:2001.05486 (Tensorflow)
"""

class Transform(nn.Module):
    """Base class for all transform objects."""
    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise NotImplementedError()


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_absdet = torch.ones(batch_size)
        for func in funcs:
            outputs, absdet = func(outputs, context)
            total_absdet *= absdet
        return outputs, total_absdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)

class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.
        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)
