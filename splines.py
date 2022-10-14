import math

import torch
from torch.nn import functional as F
import numpy as np

import utils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_EPS = 1e-5
DEFAULT_QUADRATIC_THRESHOLD = 1e-3

"""
Free inspiration from :
    https://github.com/bayesiains/nsf, arXiv:1906.04032 (PyTorch)
    https://gitlab.com/i-flow/i-flow/, arXiv:2001.05486 (Tensorflow)
"""

def linear_spline(inputs, unnormalized_pdf,
                  inverse=False,
                  left=0., right=1., bottom=0., top=1.):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    if not inverse and (torch.min(inputs) < left or torch.max(inputs) > right):
        raise RuntimeError("Input of forward linear spline outside of domain")
    elif inverse and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        raise RuntimeError("Input of inverse linear spline outside of domain")
    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.
    cdf = F.pad(cdf, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        inv_bin_idx = utils.searchsorted(cdf, inputs)

        bin_boundaries = (torch.linspace(0, 1, num_bins+1)
                          .view([1] * inputs.dim() + [-1])
                          .expand(*inputs.shape, -1))

        slopes = ((cdf[..., 1:] - cdf[..., :-1])
                  / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1]))
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1

        alpha = bin_pos - bin_idx.float()

        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = torch.log(input_pdfs) - np.log(bin_width)

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)
    absdet = torch.exp(-logabsdet)
    return outputs, absdet

def quadratic_spline(inputs,
                     unnormalized_widths,
                     unnormalized_heights,
                     inverse=False,
                     left=0., right=1., bottom=0., top=1.,
                     min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                     min_bin_height=DEFAULT_MIN_BIN_HEIGHT):

    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    if not inverse and (torch.min(inputs) < left or torch.max(inputs) > right):
        raise transforms.InputOutsideDomain()
    elif inverse and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        raise transforms.InputOutsideDomain()

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise RuntimeError("Input of forward quadratic spline outside of domain")
    if min_bin_height * num_bins > 1.0:
        raise RuntimeError("Input of inverse quadratic spline outside of domain")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    unnorm_heights_exp = torch.exp(unnormalized_heights)

    if unnorm_heights_exp.shape[-1] == num_bins - 1:
        # Set boundary heights s.t. after normalization they are exactly 1.
        first_widths = 0.5 * widths[..., 0]
        last_widths = 0.5 * widths[..., -1]
        numerator = (0.5 * first_widths * unnorm_heights_exp[..., 0]
                     + 0.5 * last_widths * unnorm_heights_exp[..., -1]
                     + torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                 * widths[..., 1:-1], dim=-1))
        constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
        constant = constant[..., None]
        unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)

    unnormalized_area = torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                  * widths, dim=-1)[..., None]
    heights = unnorm_heights_exp / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights

    bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
    bin_left_cdf[..., -1] = 1.
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode='constant', value=0.0)

    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        bin_idx = utils.searchsorted(bin_left_cdf, inputs)[..., None]
    else:
        bin_idx = utils.searchsorted(bin_locations, inputs)[..., None]

    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]

    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx+1)[..., 0]

    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c_)) / (2*a)
        outputs = alpha * input_bin_widths + input_bin_locations
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = -torch.log((alpha * (input_right_heights - input_left_heights)
                                + input_left_heights))
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha.pow(2) + b * alpha + c
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = torch.log((alpha * (input_right_heights - input_left_heights)
                               + input_left_heights))

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)

    absdet = torch.exp(-logabsdet)
    return outputs, absdet

def cubic_spline(inputs,
                 unnormalized_widths,
                 unnormalized_heights,
                 unnorm_derivatives_left,
                 unnorm_derivatives_right,
                 inverse=False,
                 left=0., right=1., bottom=0., top=1.,
                 min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                 eps=DEFAULT_EPS,
                 quadratic_threshold=DEFAULT_QUADRATIC_THRESHOLD):
    """
    References:
    > Blinn, J. F. (2007). How to solve a cubic equation, part 5: Back to numerics. IEEE Computer
    Graphics and Applications, 27(3):78–89.
    """
    if not inverse and (torch.min(inputs) < left or torch.max(inputs) > right):
        raise transforms.InputOutsideDomain()
    elif inverse and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        raise transforms.InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths[..., -1] = 1
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    cumheights = torch.cumsum(heights, dim=-1)
    cumheights[..., -1] = 1
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)

    slopes = heights / widths
    min_something_1 = torch.min(torch.abs(slopes[..., :-1]),
                                torch.abs(slopes[..., 1:]))
    min_something_2 = (
            0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
            / (widths[..., :-1] + widths[..., 1:])
    )
    min_something = torch.min(min_something_1, min_something_2)

    derivatives_left = torch.sigmoid(unnorm_derivatives_left) * 3 * slopes[..., 0][..., None]
    derivatives_right = torch.sigmoid(unnorm_derivatives_right) * 3 * slopes[..., -1][..., None]

    derivatives = min_something * (torch.sign(slopes[..., :-1]) + torch.sign(slopes[..., 1:]))
    derivatives = torch.cat([derivatives_left,
                             derivatives,
                             derivatives_right], dim=-1)

    a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths.pow(2)
    b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
    c = derivatives[..., :-1]
    d = cumheights[..., :-1]

    if inverse:
        bin_idx = utils.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = utils.searchsorted(cumwidths, inputs)[..., None]

    inputs_a = a.gather(-1, bin_idx)[..., 0]
    inputs_b = b.gather(-1, bin_idx)[..., 0]
    inputs_c = c.gather(-1, bin_idx)[..., 0]
    inputs_d = d.gather(-1, bin_idx)[..., 0]

    input_left_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_right_cumwidths = cumwidths.gather(-1, bin_idx + 1)[..., 0]

    if inverse:
        # Modified coefficients for solving the cubic.
        inputs_b_ = (inputs_b / inputs_a) / 3.
        inputs_c_ = (inputs_c / inputs_a) / 3.
        inputs_d_ = (inputs_d - inputs) / inputs_a

        delta_1 = -inputs_b_.pow(2) + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_.pow(2)

        discriminant = 4. * delta_1 * delta_3 - delta_2.pow(2)

        depressed_1 = -2. * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0  # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        outputs = torch.zeros_like(inputs)

        # Deal with one root cases.

        p = utils.cbrt((-depressed_1[one_root_mask] + torch.sqrt(-discriminant[one_root_mask])) / 2.)
        q = utils.cbrt((-depressed_1[one_root_mask] - torch.sqrt(-discriminant[one_root_mask])) / 2.)

        outputs[one_root_mask] = ((p + q)
                                  - inputs_b_[one_root_mask]
                                  + input_left_cumwidths[one_root_mask])

        # Deal with three root cases.

        theta = torch.atan2(torch.sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = torch.cos(theta)
        cubic_root_2 = torch.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * math.sqrt(3) * cubic_root_2
        root_3 = -0.5 * cubic_root_1 + 0.5 * math.sqrt(3) * cubic_root_2

        root_scale = 2 * torch.sqrt(-depressed_2[three_roots_mask])
        root_shift = (-inputs_b_[three_roots_mask] + input_left_cumwidths[three_roots_mask])

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_1).float()
        root1_mask *= (root_1 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root2_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_2).float()
        root2_mask *= (root_2 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        root3_mask = ((input_left_cumwidths[three_roots_mask] - eps) < root_3).float()
        root3_mask *= (root_3 < (input_right_cumwidths[three_roots_mask] + eps)).float()

        roots = torch.stack([root_1, root_2, root_3], dim=-1)
        masks = torch.stack([root1_mask, root2_mask, root3_mask], dim=-1)
        mask_index = torch.argsort(masks, dim=-1, descending=True)[..., 0][..., None]
        outputs[three_roots_mask] = torch.gather(roots, dim=-1, index=mask_index).view(-1)

        # Deal with a -> 0 (almost quadratic) cases.

        quadratic_mask = inputs_a.abs() < quadratic_threshold
        a = inputs_b[quadratic_mask]
        b = inputs_c[quadratic_mask]
        c = (inputs_d[quadratic_mask] - inputs[quadratic_mask])
        alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c)) / (2*a)
        outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]

        shifted_outputs = (outputs - input_left_cumwidths)
        logabsdet = -torch.log((3 * inputs_a * shifted_outputs.pow(2) +
                                2 * inputs_b * shifted_outputs +
                                inputs_c))
    else:
        shifted_inputs = (inputs - input_left_cumwidths)
        outputs = (inputs_a * shifted_inputs.pow(3) +
                   inputs_b * shifted_inputs.pow(2) +
                   inputs_c * shifted_inputs +
                   inputs_d)

        logabsdet = torch.log((3 * inputs_a * shifted_inputs.pow(2) +
                               2 * inputs_b * shifted_inputs +
                               inputs_c))

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)

    absdet = torch.exp(-logabsdet)
    return outputs, absdet
