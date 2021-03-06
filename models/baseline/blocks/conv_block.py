from torch import nn


class ConvBlock(nn.Module):
    """Convolution with batch norm, activation, and dropout."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 batch_norm=True,
                 dropout=False,
                 act_fn=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              padding_mode=padding_mode)

        if batch_norm is True:
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.batch_norm = None

        if dropout is True:
            self.dropout = nn.Dropout2d(0.3)
        else:
            self.dropout = None

        if act_fn == 'relu':
            self.act_fn = nn.ReLU(inplace=True)

        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()

        else:
            self.act_fn = None

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = self.conv(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.act_fn is not None:
            x = self.act_fn(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class DoubleConvBlock(nn.Module):
    """Double convolution block."""

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 batch_norm=True,
                 dropout=False,
                 act_fn=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(DoubleConvBlock, self).__init__()

        self.conv_block_1 = ConvBlock(in_channels,
                                      mid_channels,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn=act_fn,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.conv_block_2 = ConvBlock(mid_channels,
                                      out_channels,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn=act_fn,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

    # pylint: disable=arguments-differ
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        return x
