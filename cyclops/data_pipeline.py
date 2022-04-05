"""Data-pipeline manager module."""

import logging

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class DataPipelineManager:
    """A manager used to run SQL extraction, Processing."""

    def __init__(self, config):
        """Instantiate."""

    def run(self):
        """Run pipeline SQL querier -> Processor -> data."""

    def setup(self):
        """Set up data pipeline manager."""
