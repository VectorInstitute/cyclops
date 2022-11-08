"""Test logging functions."""

import logging

from cyclops.utils.log import setup_logging


def test_logger():
    """Test logger."""
    logger = logging.getLogger(__name__)
    setup_logging(print_level="INFO", logger=logger)
    setup_logging(print_level="INFO", logger=logger, use_color=False)
