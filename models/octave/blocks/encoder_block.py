"""
Module of an encoder block.
"""

from torch import nn

from models.octave.blocks.octave_conv_block import DoubleOctaveConvBlock
from models.octave.blocks.octave_conv_block import OctaveConvBlock
from models.octave.layers.octave_pool2d import OctaveAvgPool2d
from models.octave.layers.octave_pool2d import OctaveMaxPool2d


class EncoderBlock(nn.Module):
    """
    Encoder block of down sample and double octave convolution.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 in_alpha, mid_alpha, out_alpha,
                 downsample='avg', scale_factor=2,
                 batch_norm=True, dropout=False, act_fn=None,
                 spatial_ratio=2, merge_mode='padding',
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', **kwargs):
        super(EncoderBlock, self).__init__()

        if downsample == 'max':
            self.downsample = OctaveMaxPool2d(in_alpha, scale_factor)

        elif downsample == 'avg':
            self.downsample = OctaveAvgPool2d(in_alpha, scale_factor)

        elif downsample == 'conv':
            self.downsample = OctaveConvBlock(in_channels, mid_channels,
                                              in_alpha, mid_alpha,
                                              batch_norm=True,
                                              dropout=False,
                                              act_fn='relu',
                                              spatial_ratio=2,
                                              merge_mode='padding',
                                              kernel_size=3,
                                              stride=scale_factor,
                                              padding=1,
                                              dilation=1,
                                              groups=1,
                                              bias=True,
                                              padding_mode='zeros',
                                              **kwargs)

        else:
            raise NotImplementedError

        self.double_conv = DoubleOctaveConvBlock(in_channels, mid_channels,
                                                 out_channels, in_alpha,
                                                 mid_alpha, out_alpha,
                                                 batch_norm=batch_norm,
                                                 dropout=dropout,
                                                 act_fn=act_fn,
                                                 spatial_ratio=spatial_ratio,
                                                 merge_mode=merge_mode,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation,
                                                 groups=groups,
                                                 bias=bias,
                                                 padding_mode=padding_mode,
                                                 **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        input_h, input_l = self.downsample(input_h, input_l)
        input_h, input_l = self.double_conv(input_h, input_l)
        return input_h, input_l
