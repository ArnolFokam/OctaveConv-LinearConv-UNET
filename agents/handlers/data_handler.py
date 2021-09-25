"""Agent for handling data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from agents.handlers.configs_handler import ConfigsHandler
from agents.handlers.paths_handler import PathsHandler
from datasets.brain_lesion_segmentation import BrainLesionSegmentationDataset
from processings.dataloaders import data_loaders

LOGGER = logging.getLogger(__name__)


class DataHandler(PathsHandler, ConfigsHandler):
    """Agent for handling data."""

    def __init__(self, external_configs_list: list = None):
        # init configs
        super(DataHandler, self).__init__(external_configs_list)

        # init sample keys
        self.image_key = None
        self.target_key = None
        self.mask_key = None

        # data loaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def handle_data(self):
        """Get data loaders and sample keys."""
        # self.image_key, self.mask_key = self.get_sample_keys()
        (self.train_loader, self.valid_loader,
         self.test_loader) = self.get_data_loaders()

    def get_sample_keys(self) -> (str, str, str):
        """Get sample keys for reading data of image, target, and mask."""
        # get keys of data sample
        image_key = self.configs.DATA.DATASET.IMAGE_KEY
        mask_key = self.configs.DATA.DATASET.MASK_KEY

        return image_key, mask_key

    @staticmethod
    def get_data_shape(data_loader: torch.utils.data.DataLoader,
                       data_key: str) -> [int, int, int, int]:
        """Get data shape from data loader."""
        data_shape = next(iter(data_loader))[data_key].shape
        return data_shape

    def get_data_loaders(self) -> (torch.utils.data.DataLoader,
                                   torch.utils.data.DataLoader,
                                   torch.utils.data.DataLoader):
        """Get custom train, valid, and test data loader."""
        workers = self.configs.DATA.DATASET.WORKERS
        batch_size = self.configs.DATA.BATCH_SIZE
        image_size = self.configs.DATA.DATASET.IMAGE_SIZE
        aug_scale = self.configs.DATA.DATASET.AUGMENTATION.SCALE
        aug_angle = self.configs.DATA.DATASET.AUGMENTATION.ANGLE
        dataset_dir = self.configs.DATA.DATASET.DATASET_DIR

        train, val, test = data_loaders(BrainLesionSegmentationDataset,
                                        dataset_dir,
                                        batch_size,
                                        workers,
                                        image_size,
                                        aug_scale,
                                        aug_angle)

        return train, val, test
