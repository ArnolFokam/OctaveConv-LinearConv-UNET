from abc import ABC

from agents.handlers.output_handler import OutputHandler
from agents.handlers.state_handlers import StateHandler
from agents.handlers.summary_handler import SummaryHandler
from agents.segmentation.utils.get_model import get_model


class SegmentationAgent(SummaryHandler, StateHandler, OutputHandler, ABC):
    def __init__(self, external_configs_list: list = None):
        super(SegmentationAgent, self).__init__()

        # setup paths, make sure the paths exists
        # and backup existing config files
        self.handle_paths()

        # get sample keys and data loaders
        self.handle_data()

    def get_model(self):
        """Get model."""
        return get_model(self.configs)
