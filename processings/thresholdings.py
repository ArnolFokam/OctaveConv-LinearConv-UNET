"""Thresholding operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import torch

from skimage import filters
from PIL import Image

from utils.helpers import convert_to_ndarray

LOGGER = logging.getLogger(__name__)

# thresh methods from skimage returns threshold_value_maps
THRESHOLD_METHODS = {
    'otsu': filters.threshold_otsu,
    'isodata': filters.threshold_isodata,
    'li': filters.threshold_li,
    'mean': filters.threshold_mean,
    'niblack': filters.threshold_niblack,
    'sauvola': filters.threshold_sauvola,
    'triangle': filters.threshold_triangle,
    'yen': filters.threshold_yen,
    'local': filters.threshold_local
}


def batch_thresholding(prob_maps, thresh_mode: str = 'constant',
                       **kwargs):
    """Apply global thresholding on a probability maps."""
    if isinstance(prob_maps, (np.ndarray, torch.Tensor)):
        case_a = bool(prob_maps.ndim == 4 and prob_maps.shape[1] == 1)
        case_b = bool(prob_maps.ndim == 3)
        assert case_a or case_b, 'Probablity maps should be "N1HW" or "NHW"'
        prob_maps = prob_maps.squeeze(1) if case_a is True else prob_maps

        # skimage methods work on numpy ndarrays
        if isinstance(prob_maps, torch.Tensor):
            prob_maps = convert_to_ndarray(prob_maps)

    elif isinstance(prob_maps, Image.Image):
        prob_maps = np.array(prob_maps)[None, :, :]

    else:
        raise NotImplementedError

    num_samples = len(prob_maps)
    if thresh_mode in ('local', 'niblack', 'sauvola'):
        threshold_values = np.zeros_like(prob_maps)

    else:
        threshold_values = np.zeros(num_samples)

    for idx, prob_map in enumerate(prob_maps):
        if thresh_mode == 'constant':
            threshold_values[idx] = kwargs.get('constant', 0.5)
        else:
            thresholding = THRESHOLD_METHODS[thresh_mode]
            if thresh_mode == 'local':
                threshold_values[idx] = thresholding(
                    prob_map, kwargs.get('block_size', 5))

            else:
                threshold_values[idx] = thresholding(prob_map)

    if thresh_mode in ('local', 'niblack', 'sauvola'):
        binary_maps = (prob_maps > threshold_values).astype(float)

    else:
        binary_maps = (prob_maps > np.reshape(
            threshold_values, (num_samples, 1, 1))).astype(float)

    binary_maps = binary_maps[:, None, :, :]

    return binary_maps