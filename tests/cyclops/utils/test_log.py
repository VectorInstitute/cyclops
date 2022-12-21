"""Test logging functions."""

import logging
import os

from cyclops.utils.log import setup_logging


def test_logger():
    """Test logger."""
    logger = logging.getLogger(__name__)
    setup_logging(print_level="INFO", logger=logger)
    setup_logging(print_level="INFO", logger=logger, use_color=False)
    setup_logging(print_level="INFO", logger=logger, log_path="log.log")
    assert os.path.isfile("log.log")
    os.remove("log.log")
