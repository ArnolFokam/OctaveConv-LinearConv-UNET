"""
Octave UNet models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

from models.octave.blocks.initial_block import InitialBlock
from models.octave.blocks.encoder_block import EncoderBlock
from models.octave.blocks.decoder_block import DecoderBlock
from models.octave.blocks.final_block import FinalBlock