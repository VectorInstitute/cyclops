"""Data-pipeline manager module."""

import logging

from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


class DataPipelineManager:
    """A manager used to run SQL extraction, Processing."""

    def __init__(self, config):
        """Instantiate."""
        pass

    def run(self):
        """Run pipeline SQL querier -> Processor -> data."""
        pass
