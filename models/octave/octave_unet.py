"""
Octave UNet models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.octave.blocks.initial_block import InitialBlock
from models.octave.blocks.encoder_block import EncoderBlock
from models.octave.blocks.decoder_block import DecoderBlock
from models.octave.blocks.final_block import FinalBlock
from models.unet import UNetBackBone


class OctaveUNet(UNetBackBone):
    """Octave UNet with fixed number of layers."""

    def __init__(self,
                 channels=None,
                 alphas=None,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 batch_norm=True,
                 dropout=False,
                 padding_mode='zeros',
                 merge_mode='padding'):
        super(OctaveUNet, self).__init__(
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            batch_norm,
            dropout,
            padding_mode,
            merge_mode)

        if alphas is None:
            alphas = [0.5] * 6

        if self.channels is None:
            self.channels = [3, 32, 64, 128, 256, 512]
        assert len(self.channels) == 6
        assert len(alphas) == 6

        self.alphas = alphas

        encoder_input_channels = self.channels[1:-1]
        encoder_output_channels = self.channels[2:-1] + self.channels[-2:-1]
        encoder_input_alphas = alphas[1:-1]
        encoder_output_alphas = alphas[2:-1] + alphas[-2:-1]

        decoder_input_channels = self.channels[-2:0:-1]
        decoder_output_channels = self.channels[-3:0:-1] + self.channels[1:2]
        decoder_input_alphas = alphas[-2:0:-1]
        decoder_output_alphas = alphas[-3:0:-1] + alphas[1:2]

        self.add_module('encoder_0',
                        InitialBlock(in_channels=self.channels[0],
                                     mid_channels=self.channels[1],
                                     out_channels=self.channels[1],
                                     in_alpha=alphas[0],
                                     mid_alpha=alphas[1],
                                     out_alpha=alphas[1],
                                     batch_norm=self.batch_norm,
                                     dropout=self.dropout,
                                     act_fn='relu',
                                     spatial_ratio=2,
                                     merge_mode=merge_mode,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     groups=self.groups,
                                     bias=self.bias,
                                     padding_mode=self.padding_mode))

        i = 0
        for input_channel, output_channel, input_alpha, output_alpha in zip(
                encoder_input_channels, encoder_output_channels,
                encoder_input_alphas, encoder_output_alphas):
            i += 1
            self.add_module('encoder_{}'.format(i),
                            EncoderBlock(in_channels=input_channel,
                                         mid_channels=output_channel,
                                         out_channels=output_channel,
                                         in_alpha=input_alpha,
                                         mid_alpha=output_alpha,
                                         out_alpha=output_alpha,
                                         downsample='avg',
                                         scale_factor=2,
                                         batch_norm=self.batch_norm,
                                         dropout=self.dropout,
                                         act_fn='relu',
                                         spatial_ratio=2,
                                         merge_mode=merge_mode,
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups,
                                         bias=self.bias,
                                         padding_mode=self.padding_mode))

        i = len(decoder_input_channels)
        for input_channel, output_channel, input_alpha, output_alpha in zip(
                decoder_input_channels, decoder_output_channels,
                decoder_input_alphas, decoder_output_alphas):
            self.add_module('decoder_{}'.format(i),
                            DecoderBlock(in_channels=input_channel,
                                         mid_channels=output_channel,
                                         out_channels=output_channel,
                                         in_alpha=input_alpha,
                                         mid_alpha=output_alpha,
                                         out_alpha=output_alpha,
                                         upsample='transp',
                                         scale_factor=2,
                                         batch_norm=self.batch_norm,
                                         dropout=self.dropout,
                                         act_fn='relu',
                                         spatial_ratio=2,
                                         merge_mode=merge_mode,
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding=self.padding,
                                         output_padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups,
                                         bias=self.bias,
                                         padding_mode=self.padding_mode))
            i -= 1

        self.add_module('decoder_0',
                        FinalBlock(in_channels=self.channels[1],
                                   out_channels=self.channels[-1],
                                   in_alpha=alphas[1],
                                   out_alpha=alphas[-1],
                                   batch_norm=self.batch_norm,
                                   dropout=self.dropout,
                                   act_fn=None,
                                   spatial_ratio=2,
                                   merge_mode=merge_mode,
                                   kernel_size=1,
                                   stride=self.stride,
                                   padding=0,
                                   dilation=self.dilation,
                                   groups=self.groups,
                                   bias=self.bias,
                                   padding_mode=self.padding_mode))

        # pylint: disable=arguments-differ

    def forward(self, inputs):

        for name, module in self.named_children():
            if name == 'encoder_0':
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(inputs)

            elif name == 'decoder_0':
                outputs = module(locals()[name[:-1] + '1_h'],
                                 locals()[name[:-1] + '1_l'])

            elif name == 'decoder_{}'.format(len(self.channels) - 2):
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(
                    locals()['en' + name[2:] + '_h'],
                    locals()['en' + name[2:-1] +
                             str(int(name[-1]) - 1) + '_h'],
                    locals()['en' + name[2:] + '_l'],
                    locals()['en' + name[2:-1] + str(
                        int(name[-1]) - 1) + '_l'])

            elif 'encoder_' in name:
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(
                    locals()[name[:-1] + str(int(name[-1]) - 1) + '_h'],
                    locals()[name[:-1] + str(int(name[-1]) - 1) + '_l'])

            elif 'decoder_' in name:
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(
                    locals()[name[:-1] + str(int(name[-1]) + 1) + '_h'],
                    locals()['en' + name[2:-1] +
                             str(int(name[-1]) - 1) + '_h'],
                    locals()[name[:-1] + str(int(name[-1]) + 1) + '_l'],
                    locals()['en' + name[2:-1] + str(
                        int(name[-1]) - 1) + '_l'])

        return outputs
