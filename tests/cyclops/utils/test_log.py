"""Test logging functions."""

import logging

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging


def test_logger():
    """Test logger."""
    logger = logging.getLogger(__name__)
    setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=logger)
    setup_logging(
        log_path=get_log_file_path(), print_level="INFO", logger=logger, use_color=False
    )
