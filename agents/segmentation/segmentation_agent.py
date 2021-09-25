from abc import ABC

from agents.handlers.output_handler import OutputHandler
from agents.handlers.state_handlers import StateHandler
from agents.handlers.summary_handler import SummaryHandler
from agents.segmentation.utils.get_model import get_model

# Thank to https://github.com/JiajieMo/OctaveUNet/blob/master/src/agents/retina/retinal_vessel.py


class SegmentationAgent(SummaryHandler, StateHandler, OutputHandler, ABC):
    def __init__(self, external_configs_list: list = None):
        super(SegmentationAgent, self).__init__()

        # setup paths, make sure the paths exists
        # and backup existing config files
        self.handle_paths()

        # get sample keys and data loaders
        # self.handle_data()

        # get device and setup random seed and benchmark
        self.handle_device()

        # get model, criterion, optimizer, and lr_scheduler
        self.handle_computation_graph()

    def get_model(self):
        """Get model."""
        return get_model(self.configs)
