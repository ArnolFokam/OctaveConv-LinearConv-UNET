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

        self.encoder_0 = InitialBlock(in_channels=channels[0],
                                      mid_channels=channels[1],
                                      out_channels=channels[1],
                                      in_alpha=alphas[0],
                                      mid_alpha=alphas[1],
                                      out_alpha=alphas[1],
                                      batch_norm=self.batch_norm,
                                      dropout=self.self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.encoder_1 = EncoderBlock(in_channels=channels[1],
                                      mid_channels=channels[2],
                                      out_channels=channels[2],
                                      in_alpha=alphas[1],
                                      mid_alpha=alphas[2],
                                      out_alpha=alphas[2],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.encoder_2 = EncoderBlock(in_channels=channels[2],
                                      mid_channels=channels[3],
                                      out_channels=channels[3],
                                      in_alpha=alphas[2],
                                      mid_alpha=alphas[3],
                                      out_alpha=alphas[3],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.encoder_3 = EncoderBlock(in_channels=channels[3],
                                      mid_channels=channels[4],
                                      out_channels=channels[4],
                                      in_alpha=alphas[3],
                                      mid_alpha=alphas[4],
                                      out_alpha=alphas[4],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.encoder_4 = EncoderBlock(in_channels=channels[4],
                                      mid_channels=channels[4],
                                      out_channels=channels[4],
                                      in_alpha=alphas[4],
                                      mid_alpha=alphas[4],
                                      out_alpha=alphas[4],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.decoder_4 = DecoderBlock(in_channels=channels[4],
                                      mid_channels=channels[3],
                                      out_channels=channels[3],
                                      in_alpha=alphas[4],
                                      mid_alpha=alphas[3],
                                      out_alpha=alphas[3],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.decoder_3 = DecoderBlock(in_channels=channels[3],
                                      mid_channels=channels[2],
                                      out_channels=channels[2],
                                      in_alpha=alphas[3],
                                      mid_alpha=alphas[2],
                                      out_alpha=alphas[2],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.decoder_2 = DecoderBlock(in_channels=channels[2],
                                      mid_channels=channels[1],
                                      out_channels=channels[1],
                                      in_alpha=alphas[2],
                                      mid_alpha=alphas[1],
                                      out_alpha=alphas[1],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.decoder_1 = DecoderBlock(in_channels=channels[1],
                                      mid_channels=channels[1],
                                      out_channels=channels[1],
                                      in_alpha=alphas[1],
                                      mid_alpha=alphas[1],
                                      out_alpha=alphas[1],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=self.merge_mode,
                                      kernel_size=self.kernel_size,
                                      stride=self.stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      groups=self.groups,
                                      bias=self.bias,
                                      padding_mode=self.padding_mode)

        self.decoder_0 = FinalBlock(in_channels=channels[1],
                                    out_channels=channels[-1],
                                    in_alpha=alphas[1],
                                    out_alpha=alphas[-1],
                                    batch_norm=self.batch_norm,
                                    dropout=self.dropout,
                                    act_fn=None,
                                    spatial_ratio=2,
                                    merge_mode=self.merge_mode,
                                    kernel_size=1,
                                    stride=self.stride,
                                    padding=0,
                                    dilation=self.dilation,
                                    groups=self.groups,
                                    bias=self.bias,
                                    padding_mode=self.padding_mode)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        encoder_0_h, encoder_0_l = self.encoder_0(inputs)
        encoder_1_h, encoder_1_l = self.encoder_1(encoder_0_h, encoder_0_l)
        encoder_2_h, encoder_2_l = self.encoder_2(encoder_1_h, encoder_1_l)
        encoder_3_h, encoder_3_l = self.encoder_3(encoder_2_h, encoder_2_l)
        encoder_4_h, encoder_4_l = self.encoder_4(encoder_3_h, encoder_3_l)
        decoder_4_h, decoder_4_l = self.decoder_4(encoder_4_h, encoder_3_h,
                                                  encoder_4_l, encoder_3_l)
        decoder_3_h, decoder_3_l = self.decoder_3(decoder_4_h, encoder_2_h,
                                                  decoder_4_l, encoder_2_l)
        decoder_2_h, decoder_2_l = self.decoder_2(decoder_3_h, encoder_1_h,
                                                  decoder_3_l, encoder_1_l)
        decoder_1_h, decoder_1_l = self.decoder_1(decoder_2_h, encoder_0_h,
                                                  decoder_2_l, encoder_0_l)
        outputs = self.decoder_0(decoder_1_h, decoder_1_l)

        return outputs
