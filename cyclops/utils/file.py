"""Utility functions for saving/loading files."""

import logging
import os

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def exchange_extension(file_path: str, new_ext: str) -> str:
    """Exchange one file extension for another.

    Parameters
    ----------
    file_path: str
        File path in which to exchange the extension.
    new_ext: str
        New extension to replace the existing extension.

    Returns
    -------
    str
        File path with the new extension.

    """
    # Remove a leading dot
    new_ext = new_ext.strip(".")
    _, old_ext = os.path.splitext(file_path)
    return file_path[: -len(old_ext)] + "." + new_ext


def process_file_save_path(
    save_path: str, file_format: str, create_dir: bool = True
) -> str:
    """Process file save path, perform checks, and possibly create a parent directory.

    Parameters
    ----------
    save_path: str
        Path where the file will be saved.
    file_format: str
        File format of the file to save.
    create_dir: bool
        If True, create the parent directory path if needed.

    Returns
    -------
    str
        The processed save path.

    """
    # Create the directory if it doesn't already exist.
    directory, _ = os.path.split(save_path)

    if create_dir:
        # Ignore checking local paths
        if directory != "":
            os.makedirs(directory, exist_ok=True)

    # Add the .parquet extension if it isn't there.
    _, ext = os.path.splitext(save_path)

    if ext == "":
        save_path = save_path + "." + file_format
    elif ext != "." + file_format:
        raise ValueError(
            f"""The file extension on the save path must be {file_format}.
            Alternatively, consider changing the file format."""
        )

    return save_path


def save_dataframe(
    dataframe: pd.DataFrame,
    save_path: str,
    file_format: str = "parquet",
) -> str:
    """Save a DataFrame object to file.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Dataframe to save.
    save_path: str
        Path where the file will be saved.
    file_format: str
        File format of the file to save.

    Returns
    -------
    str
        Processed save path for upstream use.

    """
    save_path = process_file_save_path(save_path, file_format)

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input data is not a DataFrame.")

    LOGGER.info("Saving dataframe to %s", save_path)

    if file_format == "parquet":
        dataframe.to_parquet(save_path)
    elif file_format == "csv":
        dataframe.to_csv(save_path)
    else:
        raise ValueError(
            "Invalid file formated provided. Currently supporting 'parquet' and 'csv'."
        )

    return save_path
