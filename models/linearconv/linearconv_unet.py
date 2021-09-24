import torch

from models.linearconv.blocks.conv_block import LinearConvBlock, DoubleLinearConvBlock
from models.linearconv.blocks.decoder_block import DecoderBlock
from models.linearconv.blocks.encoder_block import EncoderBlock
from models.unet import UNetBackBone


class LinearConvUNet(UNetBackBone):

    def __init__(self,
                 channels=None,
                 variants=None,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 batch_norm=True,
                 dropout=False,
                 padding_mode='zeros',):
        super(LinearConvUNet, self).__init__(
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            batch_norm,
            dropout,
            padding_mode)

        encoder_variants = variants[1:-1]
        encoder_input_channels = self.channels[1:-1]
        encoder_output_channels = self.channels[2:-1] + self.channels[-2:-1]

        decoder_input_channels = self.channels[-2:0:-1]
        decoder_output_channels = self.channels[-3:0:-1] + self.channels[1:2]

        self.add_module("encoder_0",
                        DoubleLinearConvBlock(in_channels=self.channels[0],
                                              mid_channels=self.channels[1],
                                              out_channels=self.channels[1],
                                              batch_norm=self.batch_norm,
                                              dropout=self.dropout,
                                              act_fn='relu',
                                              kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              groups=self.groups,
                                              bias=self.bias,
                                              padding_mode=self.padding_mode,
                                              variant=variants[0]))

        i = 0
        for input_channel, output_channel, variant in zip(encoder_input_channels,
                                                          encoder_output_channels,
                                                          encoder_variants):
            i += 1
            self.add_module('encoder_{}'.format(i),
                            EncoderBlock(in_channels=input_channel,
                                         mid_channels=output_channel,
                                         out_channels=output_channel,
                                         downsample='max',
                                         scale_factor=2,
                                         batch_norm=self.batch_norm,
                                         dropout=self.dropout,
                                         act_fn='relu',
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding=self.padding,
                                         dilation=self.dilation,
                                         groups=self.groups,
                                         bias=self.bias,
                                         padding_mode=self.padding_mode,
                                         variant=variant))

        i = len(decoder_input_channels)
        for input_channel, output_channel in zip(
                decoder_input_channels, decoder_output_channels):
            self.add_module('decoder_{}'.format(i),
                            DecoderBlock(in_channels=input_channel,
                                         mid_channels=output_channel,
                                         out_channels=output_channel,
                                         upsample='transp',
                                         scale_factor=2,
                                         batch_norm=self.batch_norm,
                                         dropout=self.dropout,
                                         act_fn='relu',
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
                        LinearConvBlock(in_channels=self.channels[1],
                                        out_channels=self.channels[-1],
                                        batch_norm=self.batch_norm,
                                        dropout=self.dropout,
                                        act_fn=None,
                                        kernel_size=1,
                                        stride=self.stride,
                                        padding=0,
                                        dilation=self.dilation,
                                        groups=self.groups,
                                        bias=self.bias,
                                        padding_mode=self.padding_mode))

    def forward(self, inputs):

        for name, module in self.named_children():
            if name == 'encoder_0':
                locals()[name] = module(inputs)

            elif name == 'decoder_0':
                # merge last decoder with
                # previous decoder 'decoder_1'
                outputs = module(locals()[name[:-1] + '1'])

            elif name == 'decoder_{}'.format(len(self.channels) - 2):
                locals()[name] = module(
                    locals()['en' + name[2:]],
                    locals()['en' + name[2:-1] +
                             str(int(name[-1]) - 1)])

            elif 'encoder_' in name:
                locals()[name] = module(
                    locals()[name[:-1] + str(int(name[-1]) - 1)])

            elif 'decoder_' in name:
                locals()[name] = module(
                    locals()[name[:-1] + str(int(name[-1]) + 1)],
                    locals()['en' + name[2:-1] +
                             str(int(name[-1]) - 1)])

        return torch.sigmoid(outputs)
