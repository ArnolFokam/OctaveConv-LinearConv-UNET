"""Utilities for counting floating point operations (FLOPs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from torch import nn

from models.linearconv.layers.linear_conv2d import LinearConv2DSimple
from models.linearconv.layers.linear_transpose_conv2d import LinearTransposeConv2DSimple

LOGGER = logging.getLogger(__name__)


def compute_flops(module, inp, out):
    """Compute FLOPs."""
    if isinstance(module, nn.Conv2d):
        module_flops = compute_Conv2d_flops(module, inp, out)

    elif isinstance(module, nn.ConvTranspose2d):
        module_flops = compute_ConvTranspose2d_flops(module, inp, out)

    elif isinstance(module, nn.BatchNorm2d):
        module_flops = compute_BatchNorm2d_flops(module, inp, out)

    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        module_flops = compute_Pool2d_flops(module, inp, out)

    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU,
                             nn.LeakyReLU)):
        module_flops = compute_ReLU_flops(module, inp)

    elif isinstance(module, nn.Upsample):
        module_flops = compute_Upsample_flops(module, inp, out)

    elif isinstance(module, nn.Linear):
        module_flops = compute_Linear_flops(module, inp, out)

    elif isinstance(module,  LinearConv2DSimple):
        module_flops = compute_LinearConv2DSimple_flops(module, inp, out)

    elif isinstance(module, LinearTransposeConv2DSimple):
        module_flops = compute_LinearTransposeConv2DSimple_flops(module, inp, out)

    else:
        module_flops = 0

    return module_flops


# pylint: disable=invalid-name
def compute_ConvTranspose2d_flops(module, inp, out):
    """Compute FLOPs of 2d transposed convolution."""
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.ConvTranspose2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4

    batch_size, in_c, in_h, in_w = inp.size()
    k_h, k_w = module.kernel_size
    out_c = out.size()[1]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * in_h * in_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_Conv2d_flops(module, inp, out):
    """Compute FLOPs of Conv2d."""
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    total_flops = total_conv_flops + bias_flops
    return total_flops


def compute_LinearConv2DSimple_flops(module, inp, out):
    """Compute FLOPs of LinearConv2dSimple."""
    # Can have multiple inputs, getting the first one
    assert isinstance(module, LinearConv2DSimple)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    k_h, k_w = module.kernel_size
    out_c, out_h, out_w = out.size()[1:]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * out_h * out_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    # get the total convolution operations
    total_conv_flops = total_conv_flops + bias_flops

    total_mul = out_c // module.times
    total_add = total_mul - 1
    total_elements = (filters_per_channel - filters_per_channel // module.times)*(in_c * k_h * k_w)
    linear_ops = (total_mul + total_add)*total_elements

    total_flops = linear_ops + total_conv_flops
    return total_flops


def compute_LinearTransposeConv2DSimple_flops(module, inp, out):
    """Compute FLOPs of 2d transposed convolution."""
    # Can have multiple inputs, getting the first one
    assert isinstance(module, LinearTransposeConv2DSimple)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4

    batch_size, in_c, in_h, in_w = inp.size()
    k_h, k_w = module.kernel_size
    out_c = out.size()[1]
    groups = module.groups

    filters_per_channel = out_c // groups
    conv_per_position_flops = k_h * k_w * in_c * filters_per_channel
    active_elements_count = batch_size * in_h * in_w

    total_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_c * active_elements_count

    # get the total convolution operations
    total_conv_flops = total_conv_flops + bias_flops

    total_mul = out_c // module.times
    total_add = total_mul - 1
    total_elements = (filters_per_channel - filters_per_channel // module.times) * (in_c * k_h * k_w)
    linear_ops = (total_mul + total_add) * total_elements

    total_flops = linear_ops + total_conv_flops
    return total_flops


def compute_BatchNorm2d_flops(module, inp, out):
    """Compute FLOPs of BatchNorm2d."""
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4
    batch_flops = np.prod(inp.shape)
    if module.affine:
        batch_flops *= 2
    return batch_flops


def compute_ReLU_flops(module, inp):
    """Compute FLOPs of ReLU."""
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU,
                               nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    active_elements_count = batch_size

    for i in inp.size()[1:]:
        active_elements_count *= i

    return active_elements_count


def compute_Pool2d_flops(module, inp, out):
    """Compute FLOPs of Pool2d."""
    assert isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d)
    assert len(inp.size()) == 4
    assert len(out.size()) == 4
    return np.prod(inp.shape)


def compute_Linear_flops(module, inp, out):
    """Compute FLOPs of fully connected Linear module."""
    assert isinstance(module, nn.Linear)
    assert len(inp.size()) == 2
    assert len(out.size()) == 2
    batch_size = inp.size()[0]
    return batch_size * inp.size()[1] * out.size()[1]


def compute_Upsample_flops(module, inp, out):
    """Compute FLOPs of Upsample."""
    assert isinstance(module, nn.Upsample)
    output_size = out[0]
    batch_size = inp.size()[0]
    output_elements_count = batch_size
    for i in output_size.shape[1:]:
        output_elements_count *= i
    return output_elements_count
