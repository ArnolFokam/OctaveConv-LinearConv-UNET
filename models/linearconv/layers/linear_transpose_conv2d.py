import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules.conv import _ConvTransposeNd
from torch.nn.modules.utils import _pair


class _LinearTransposeConv2D(_ConvTransposeNd):
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
                 output_padding=0,
                 ratio=2):
        self.kernel_size_int = kernel_size
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super(_LinearTransposeConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                                     dilation, True, output_padding, groups, bias, padding_mode)

        self.times = ratio  # default: ratio 1/2

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

    def _conv_forward(self, inputs, weight, bias, output_size=None):
        output_padding = self._output_padding(
            inputs, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        return F.conv_transpose2d(
            inputs, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class LinearTransposeConv2DSimple(_LinearTransposeConv2D):
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
                 output_padding=0,
                 ratio=2):
        super(LinearTransposeConv2DSimple, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            output_padding,
            ratio)

        self.linear_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, output_size=None):
        correlated_weights = torch.mm(self.linear_weights,
                                      self.conv_weights.reshape(self.out_channels // self.times, -1)) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias, output_size=output_size)


# Const. low-rank version
class LinearTransposeConv2DLowRank(_LinearTransposeConv2D):
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
                 output_padding=0,
                 ratio=2):
        super(LinearTransposeConv2DLowRank, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            output_padding,
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

    def forward(self, inputs, output_size=None):
        correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.conv_weights.reshape(
            self.out_channels // self.times, -1))) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias, output_size=output_size)


# Rank-ratio version
class LinearTransposeConv2DRankRatio(_LinearTransposeConv2D):
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
                 output_padding=0,
                 ratio=2):
        super(LinearTransposeConv2DRankRatio, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            output_padding,
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

    def forward(self, inputs, output_size=None):
        correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.conv_weights.reshape(
            self.out_channels // self.times, -1))) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size_int,
                     self.kernel_size_int)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias, output_size=output_size)


# Sparse version
class LinearTransposeConv2DSparse(_LinearTransposeConv2D):
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
                 output_padding=0,
                 ratio=2):
        super(LinearTransposeConv2DSparse, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            output_padding,
            ratio
        )

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

    def forward(self, inputs, output_size=None):
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
                                  bias=self.bias, output_size=output_size)
