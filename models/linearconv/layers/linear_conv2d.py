import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class _LinearConv2D(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 ratio=2):
        self.kernel_size_int = kernel_size
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super(_LinearConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode)

        self.times = ratio  # default ratio 1/2
        print(out_channels // self.times)

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)

        self.conv_weights = nn.Parameter(
            torch.Tensor(out_channels // self.times, in_channels, self.kernel_size_int, self.kernel_size_int))

        nn.init.xavier_uniform_(self.conv_weights)

    def _conv_forward(self, inputs, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(inputs, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias,
                            stride=self.stride,
                            padding=_pair(0),
                            dilation=self.dilation,
                            groups=self.groups)
        return F.conv2d(inputs,
                        weight,
                        bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class LinearConv2DSimple(_LinearConv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 ratio=2):

        super(LinearConv2DSimple, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            ratio)

        self.linear_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        correlated_weights = torch.mm(self.linear_weights,
                                      self.conv_weights.reshape(self.out_channels // self.times, -1)) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)


# Const. low-rank version
class LinearConv2DLowRank(_LinearConv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 rank=1,
                 ratio=2):

        super(LinearConv2DLowRank, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            ratio)

        self.rank = rank

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        self.column_weights = nn.Parameter(torch.Tensor(out_channels - out_channels // self.times, self.rank))
        self.row_weights = nn.Parameter(torch.Tensor(self.rank, out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.conv_weights.reshape(
            self.out_channels // self.times, -1))) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)


# Rank-ratio version
class LinearConv2DRankRatio(_LinearConv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 rank=1,
                 ratio=2):

        super(LinearConv2DRankRatio, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            ratio)

        self.rank = rank

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        self.column_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, int((out_channels // self.times) * self.rank)))
        self.row_weights = nn.Parameter(
            torch.Tensor(int((out_channels // self.times) * self.rank), out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.conv_weights.reshape(
            self.out_channels // self.times, -1))) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)


# Sparse version
class LinearConv2DSparse(_LinearConv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 stride=1,
                 groups=1,
                 padding_mode='zeros',
                 prune_step=500,
                 req_percentile=0.25,
                 thresh_step=0.00001,
                 dilation=1,
                 bias=True,
                 ratio=2):
        super(LinearConv2DSparse, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            ratio)

        self.prune_step = prune_step
        self.req_percentile = req_percentile
        self.thresh_step = thresh_step
        self.counter = 0
        self.threshold = 0

        self.linear_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, out_channels // self.times))

        self.linear_weights.data.uniform_(-0.1, 0.1)

        self.mask = nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad=False)
        self.mask.requires_grad = False
        self.percentile = 1. - float(torch.sum(self.mask).item()) / (self.mask.shape[0] ** 2)

    def forward(self, inputs):

        self.counter += 1
        if self.counter == self.prune_step:
            self.counter = 0
            self.mask = nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad=False)
            self.percentile = 1. - float(torch.sum(self.mask).item()) / (self.mask.shape[0] ** 2)
            self.threshold += (2. / (1. + 10 ** (10 * (self.percentile - self.req_percentile))) - 1) * self.thres_step
            print('pruned... %.2f, %.5f' % (self.percentile, self.threshold))

        self.mask = nn.Parameter(self.mask.type(torch.FloatTensor).to(self.device), requires_grad=False)
        temp = self.linear_weights * self.mask
        correlated_weights = torch.mm(temp, self.conv_weights.reshape(self.out_channels // self.times, -1)) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)
