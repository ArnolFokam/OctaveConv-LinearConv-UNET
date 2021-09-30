"""Get model related objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from configs.config_node import ConfigNode
from models.octave.octave_linearconv_unet import OctaveLinearConvUNet

from models.octave.octave_unet import OctaveUNet
from models.linearconv.linearconv_unet import LinearConvUNet
from models.baseline.baseline_unet import BaselineUNet


LOGGER = logging.getLogger(__name__)


def get_model(configs: ConfigNode):
    """Get model."""
    model_name = configs.MODEL.MODEL_NAME
    if model_name == 'octave':
        kwargs = {
            'channels': configs.MODEL.CHANNELS,
            'alphas': configs.MODEL.ALPHAS,
            'kernel_size': configs.MODEL.KERNEL_SIZE,
            'stride': configs.MODEL.STRIDE,
            'padding': configs.MODEL.PADDING,
            'dilation': configs.MODEL.DILATION,
            'groups': configs.MODEL.GROUPS,
            'bias': configs.MODEL.ENABLE_BIAS,
            'batch_norm': configs.MODEL.ENABLE_BATCH_NORM,
            'dropout': configs.MODEL.ENABLE_DROPOUT,
            'padding_mode': 'zeros',
            'merge_mode': 'padding',
        }

        model = OctaveUNet(**kwargs)

    elif model_name == 'linearconv':
        kwargs = {
            'channels': configs.MODEL.CHANNELS,
            'variants': configs.MODEL.VARIANTS,
            'ratios': configs.MODEL.RATIOS,
            'kernel_size': configs.MODEL.KERNEL_SIZE,
            'stride': configs.MODEL.STRIDE,
            'padding': configs.MODEL.PADDING,
            'dilation': configs.MODEL.DILATION,
            'groups': configs.MODEL.GROUPS,
            'bias': configs.MODEL.ENABLE_BIAS,
            'batch_norm': configs.MODEL.ENABLE_BATCH_NORM,
            'dropout': configs.MODEL.ENABLE_DROPOUT,
            'padding_mode': 'zeros',
        }

        model = LinearConvUNet(**kwargs)

    elif model_name == 'octavelinearconv':
        kwargs = {
            'channels': configs.MODEL.CHANNELS,
            'variants': configs.MODEL.VARIANTS,
            'alphas': configs.MODEL.ALPHAS,
            'ratios': configs.MODEL.RATIOS,
            'kernel_size': configs.MODEL.KERNEL_SIZE,
            'stride': configs.MODEL.STRIDE,
            'padding': configs.MODEL.PADDING,
            'dilation': configs.MODEL.DILATION,
            'groups': configs.MODEL.GROUPS,
            'bias': configs.MODEL.ENABLE_BIAS,
            'batch_norm': configs.MODEL.ENABLE_BATCH_NORM,
            'dropout': configs.MODEL.ENABLE_DROPOUT,
            'padding_mode': 'zeros',
        }

        model = OctaveLinearConvUNet(**kwargs)

    elif model_name == 'baseline':
        kwargs = {
            'channels': configs.MODEL.CHANNELS,
            'kernel_size': configs.MODEL.KERNEL_SIZE,
            'stride': configs.MODEL.STRIDE,
            'padding': configs.MODEL.PADDING,
            'dilation': configs.MODEL.DILATION,
            'groups': configs.MODEL.GROUPS,
            'bias': configs.MODEL.ENABLE_BIAS,
            'batch_norm': configs.MODEL.ENABLE_BATCH_NORM,
            'dropout': configs.MODEL.ENABLE_DROPOUT,
            'padding_mode': 'zeros',
        }

        model = BaselineUNet(**kwargs)
    else:
        raise NotImplemented

    LOGGER.info('Retrieved model: %s', model_name)

    return model
