"""Utility functions for saving/loading files."""

import logging
import os
from typing import Optional

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.constants import UNDERSCORE
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def save_dataframe(
    dataframe: pd.DataFrame,
    folder_path: str,
    file_name: str,
    prefix: Optional[str] = None,
) -> None:
    """Save queried data in Parquet format.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Input dataframe to save.
    folder_path: str
        Path to directory where the file can be saved.
    file_name: str
        Name of file. Extension will be .gzip.
    prefix: str, optional
        Prefix to add to file_name.

    """
    os.makedirs(folder_path, exist_ok=True)
    if prefix:
        file_name = prefix + UNDERSCORE + file_name
    save_path = os.path.join(folder_path, file_name + ".gzip")
    if isinstance(dataframe, pd.DataFrame):
        LOGGER.info("Saving dataframe to %s", save_path)
        dataframe.to_parquet(save_path)
    else:
        LOGGER.warning("Input data is not a valid dataframe, nothing to save!")
