import torch
from torch import nn

from models.baseline.blocks.conv_block import DoubleConvBlock


class DecoderBlock(nn.Module):
    """
    Decoder block of  transposed convolution and double octave convolution.
    """

    def __init__(self, in_channels,
                 mid_channels,
                 out_channels,
                 upsample='transp',
                 scale_factor=2,
                 batch_norm=True,
                 dropout=False,
                 act_fn=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 output_padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(DecoderBlock, self).__init__()

        if upsample == 'transp':
            self.upsample = nn.ConvTranspose2d(in_channels,
                                               in_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=padding,
                                               output_padding=output_padding,
                                               groups=groups,
                                               bias=bias,
                                               dilation=dilation,
                                               padding_mode=padding_mode)

        elif upsample in ('bilinear', 'nearest'):
            self.upsample = nn.Upsample(
                scale_factor=scale_factor,
                mode=upsample)

        else:
            raise NotImplementedError

        self.double_conv = DoubleConvBlock(
            in_channels*2,
            mid_channels,
            out_channels,
            batch_norm=batch_norm,
            dropout=dropout,
            act_fn=act_fn,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode
        )

    # pylint: disable=arguments-differ
    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = torch.cat((x, skip_x), dim=1)
        x = self.double_conv(x)

        return x
