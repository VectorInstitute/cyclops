"""Utility functions for processor API."""

import logging

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def is_timeseries_data(data: pd.DataFrame) -> bool:
    """Check if input data is time-series dataframe (multi-index).

    Parameters
    ----------
    data: pandas.DataFrame
        Input dataframe.

    Returns
    -------
    bool
        Yes if dataframe has multi-index (timeseries), No otherwise.

    """
    return isinstance(data.index, pd.core.indexes.multi.MultiIndex)


def check_must_have_columns(data: pd.DataFrame, must_have_columns: list) -> bool:
    """Check if data has required columns for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    must_have_columns: list
        List of column names that must be present in data.

    Returns
    -------
    bool
        True if all required columns are present, else False.

    """
    column_names = list(data.columns)
    return set(must_have_columns).issubset(column_names)


def gather_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Gather specified columns and discard rest.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    columns: list
        List of column names to keep in data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with required columns, other columns discarded.

    """
    return data[columns]


def log_counts_step(data, step_description: str, rows=True, columns=False) -> None:
    """Log num. of encounters and num. of samples (rows).

    Parameters
    ----------
    data: pandas.DataFrame
        Encounter specific input data.
    step_description: str
        Description of intermediate processing step.
    rows: bool
        Log the number of samples, or rows of data.
    columns: bool
        Log the number of data columns.

    """
    LOGGER.info(step_description)
    num_encounters = data[ENCOUNTER_ID].nunique()
    if rows:
        num_samples = len(data)
        LOGGER.info("# samples: %d, # encounters: %d", num_samples, num_encounters)
    if columns:
        num_columns = len(data.columns)
        LOGGER.info("# columns: %d, # encounters: %d", num_columns, num_encounters)
