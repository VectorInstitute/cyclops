"""Codebase related operations functions."""

import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_log_file_path() -> str:
    """Get file path to output logs.

    Returns
    -------
    str
        Path to store log file.
    """
    return os.path.join(PROJECT_ROOT, "log.log")
