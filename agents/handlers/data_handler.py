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

        # data loaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def handle_data(self):
        """Get data loaders and sample keys."""
        # self.image_key, self.mask_key = self.get_sample_keys()
        (self.train_loader, self.valid_loader,
         self.test_loader) = self.get_data_loaders()

    @staticmethod
    def get_data_shape(data_loader: torch.utils.data.DataLoader) -> [int, int, int, int]:
        """Get data shape from data loader."""
        data_shape = next(iter(data_loader))[0].shape
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
        fraction = self.configs.DATA.DATASET.FRACTION if hasattr(self.configs.DATA.DATASET, 'FRACTION') else 1.0

        train, val, test = data_loaders(BrainLesionSegmentationDataset,
                                        dataset_dir,
                                        fraction,
                                        batch_size,
                                        workers,
                                        image_size,
                                        aug_scale,
                                        aug_angle)

        return train, val, test
