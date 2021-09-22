from collections import OrderedDict

import torch
from torch import nn


class UNetBackBone(nn.Module):
    def __init__(self,
                 channels=None,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 batch_norm=True,
                 dropout=False,
                 padding_mode='zeros',
                 pooling_stride=None):
        super(UNetBackBone, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.padding_mode = padding_mode
        self.pooling_stride = pooling_stride
