import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


class LinearConv2D(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros'):

        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super(LinearConv2D, self).__init__(
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

        self.times = 2  # ratio 1/2

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)

        self.conv_weights = nn.Parameter(
            torch.Tensor(out_channels // self.times, in_channels, kernel_size, kernel_size))
        self.linear_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)

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

    def forward(self, inputs):
        correlated_weights = torch.mm(self.linear_weights,
                                      self.conv_weights.reshape(self.out_channels // self.times, -1)) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size,
                     self.kernel_size)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)

    @staticmethod
    def count_op_LinearConv2D(m, x, y):
        x = x[0]

        multiply_adds = 1

        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size, m.kernel_size
        batch_size = x.size()[0]

        out_h = y.size(2)
        out_w = y.size(3)

        # ops per output element
        # kernel_mul = kh * kw * cin
        # kernel_add = kh * kw * cin - 1
        kernel_ops = multiply_adds * kh * kw
        bias_ops = 1 if m.biasTrue is True else 0
        ops_per_element = kernel_ops + bias_ops

        # total ops
        # num_out_elements = y.numel()
        output_elements = batch_size * out_w * out_h * cout
        conv_ops = output_elements * ops_per_element * cin // 1  # m.groups=1

        # per output element
        total_mul = m.out_channels // m.times
        total_add = total_mul - 1
        num_elements = (m.out_channels - m.out_channels // m.times) * (cin * kh * kw)
        lin_ops = (total_mul + total_add) * num_elements
        total_ops = lin_ops + conv_ops
        print(lin_ops, conv_ops)

        m.total_ops = torch.Tensor([int(total_ops)])


# Const. low-rank version
class LinearConv2DLowRank(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1,
                 dilation=1,
                 padding_mode='zeros',
                 bias=True,
                 rank=1, ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(LinearConv2DLowRank, self).__init__(
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
            padding_mode
        )

        self.rank = rank
        self.times = 2  # ratio 1/2

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)

        self.conv_weights = nn.Parameter(
            torch.Tensor(out_channels // self.times, in_channels, kernel_size, kernel_size))
        self.column_weights = nn.Parameter(torch.Tensor(out_channels - out_channels // self.times, self.rank))
        self.row_weights = nn.Parameter(torch.Tensor(self.rank, out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)

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

    def forward(self, inputs):
        correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.conv_weights.reshape(
            self.out_channels // self.times, -1))) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size,
                     self.kernel_size)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)

    @staticmethod
    def count_op_LinearConv2Dlow(m, x, y):
        x = x[0]

        multiply_adds = 1

        cin = m.in_channels
        cout = m.out_channels
        kh, kw = m.kernel_size, m.kernel_size
        batch_size = x.size()[0]

        out_h = y.size(2)
        out_w = y.size(3)

        # ops per output element
        # kernel_mul = kh * kw * cin
        # kernel_add = kh * kw * cin - 1
        kernel_ops = multiply_adds * kh * kw
        bias_ops = 1 if m.biasTrue is True else 0
        ops_per_element = kernel_ops + bias_ops

        # total ops
        # num_out_elements = y.numel()
        output_elements = batch_size * out_w * out_h * cout
        conv_ops = output_elements * ops_per_element * cin // m.groups

        # per output element
        total_mul_1 = m.out_channels // m.times
        total_add_1 = total_mul_1 - 1
        num_elements_1 = m.rank * (cin * kh * kw)  # (m.out_channels - m.out_channels//m.times)
        total_mul_2 = m.rank
        total_add_2 = total_mul_2 - 1
        num_elements_2 = (m.out_channels - m.out_channels // m.times) * (
                cin * kh * kw)  # (m.out_channels - m.out_channels//m.times)
        lin_ops = (total_mul_1 + total_add_1) * num_elements_1 + (total_mul_2 + total_add_2) * num_elements_2
        total_ops = lin_ops + conv_ops
        print(lin_ops, conv_ops)

        m.total_ops = torch.Tensor([int(total_ops)])


# Rank-ratio version
class LinearConv2DRankRatio(_ConvNd):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 groups=1,
                 dilation=1,
                 padding_mode='zeros',
                 bias=True,
                 rank=1):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(LinearConv2DRankRatio, self).__init__(
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
            padding_mode
        )

        self.rank = rank
        self.times = 2  # ratio 1/2

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)

        self.conv_weights = nn.Parameter(
            torch.Tensor(out_channels // self.times, in_channels, kernel_size, kernel_size))
        self.column_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, int((out_channels // self.times) * self.rank)))
        self.row_weights = nn.Parameter(
            torch.Tensor(int((out_channels // self.times) * self.rank), out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)

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

    def forward(self, inputs):
        correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights, self.conv_weights.reshape(
            self.out_channels // self.times, -1))) \
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size,
                     self.kernel_size)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)


# Sparse version
class LinearConv2DSparse(_ConvNd):
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
                 bias=True):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(LinearConv2DSparse, self).__init__(
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
            padding_mode
        )

        self.times = 2  # ratio 1/2
        self.prune_step = prune_step
        self.req_percentile = req_percentile
        self.thresh_step = thresh_step
        self.counter = 0
        self.threshold = 0

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        # we don't need pytorch generated
        # weights from _ConvNd since we
        # initialize our own weights
        del self.weight

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)

        self.conv_weights = nn.Parameter(
            torch.Tensor(out_channels // self.times, in_channels, kernel_size, kernel_size))
        self.linear_weights = nn.Parameter(
            torch.Tensor(out_channels - out_channels // self.times, out_channels // self.times))

        nn.init.xavier_uniform_(self.conv_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)

        if self.biasTrue:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.uniform_(-0.1, 0.1)

        self.mask = nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad=False)
        self.mask.requires_grad = False
        self.percentile = 1. - float(torch.sum(self.mask).item()) / (self.mask.shape[0] ** 2)

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
            .reshape(self.out_channels - self.out_channels // self.times, self.in_channels, self.kernel_size,
                     self.kernel_size)

        return self._conv_forward(inputs,
                                  torch.cat((self.conv_weights, correlated_weights), dim=0),
                                  bias=self.bias)
