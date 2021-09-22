from torch import nn

from models.baseline.blocks.conv_block import ConvBlock, DoubleConvBlock


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 downsample='avg',
                 scale_factor=2,
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
        super(EncoderBlock, self).__init__()

        # TODO: Add parameter for stride pooling
        if downsample == 'max':
            self.downsample = nn.MaxPool2d(kernel_size=scale_factor)

        elif downsample == 'avg':
            self.downsample = nn.AvgPool2d(kernel_size=scale_factor)

        elif downsample == 'conv':
            self.downsample = ConvBlock(in_channels,
                                        mid_channels,
                                        batch_norm=True,
                                        dropout=False,
                                        act_fn='relu',
                                        kernel_size=3,
                                        stride=scale_factor,
                                        padding=1,
                                        dilation=1,
                                        groups=1,
                                        bias=True,
                                        padding_mode='zeros')

        else:
            raise NotImplementedError

        self.double_conv = DoubleConvBlock(in_channels,
                                           mid_channels,
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
        x = self.downsample(x)
        x = self.double_conv(x)
        return x
