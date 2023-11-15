"""Utility functions for processor API."""

import logging
from functools import wraps
from typing import Any, Callable, List, Optional, Union

import pandas as pd

from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def create_indicator_variables(
    features: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create binary indicator variable for each column (or specified).

    Create new indicator variable columns based on NAs for other feature columns.

    Parameters
    ----------
    features: pandas.DataFrame
        Input features with missing values.
    columns: List[str], optional
        Columns to create variables, all if not specified.

    Returns
    -------
    pandas.DataFrame
        Dataframe with indicator variables as columns.

    """
    indicator_features = features[columns] if columns else features

    return indicator_features.notnull().astype(int).add_suffix("_indicator")


def is_timestamp_series(series: pd.Series, raise_error: bool = False) -> bool:
    """Check whether a series has the Pandas Timestamp datatype.

    Parameters
    ----------
    series: pandas.Series
        A series.

    Returns
    -------
    bool
        Whether the series has the Pandas Timestamp datatype.

    """
    is_timestamp = series.dtype == pd.to_datetime(["2069-03-29 02:30:00"]).dtype
    if not is_timestamp and raise_error:
        raise ValueError(f"{series.name} must be a timestamp Series.")

    return is_timestamp


def has_columns(
    data: pd.DataFrame,
    cols: Union[str, List[str]],
    exactly: bool = False,
    raise_error: bool = False,
) -> bool:
    """Check if data has required columns for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    cols: str or list or str
        List of column names that must be present in data.
    raise_error: bool
        Whether to raise a ValueError if there are missing columns.

    Returns
    -------
    bool
        True if all required columns are present, otherwise False.

    """
    cols = to_list(cols)
    required_set = set(cols)
    columns = set(data.columns)
    present = required_set.issubset(columns)

    if not present and raise_error:
        missing = required_set - columns
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")

    if exactly:
        exact = present and len(data.columns) == len(cols)
        if not exact and raise_error:
            raise ValueError(f"Must have exactly the columns: {', '.join(cols)}.")

    return present


def has_range_index(data: pd.DataFrame) -> Union[bool, pd.Series, pd.DataFrame]:
    """Check whether a DataFrame has a range index.

    Parameters
    ----------
    data: pandas.DataFrame
        Data.

    Returns
    -------
    bool or pandas.Series or pandas.DataFrame
        Whether the data has a range index.

    """
    return (data.index == pd.RangeIndex(stop=len(data))).all()


def to_range_index(data: pd.DataFrame) -> pd.DataFrame:
    """Force a DataFrame to have a range index.

    Parameters
    ----------
    data: pandas.DataFrame
        Data.

    Returns
    -------
    pandas.DataFrame
        Data with a range index.

    """
    if has_range_index(data):
        return data

    name = data.index.name
    data = data.reset_index()
    if name == "index":
        data = data.drop("index", axis=1)

    return data


def gather_columns(data: pd.DataFrame, columns: Union[List[str], str]) -> pd.DataFrame:
    """Gather specified columns, discarding rest and return copy of columns.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    columns: list of str or str
        Column names to gather from dataframe.

    Returns
    -------
    pandas.DataFrame
        DataFrame with required columns, other columns discarded.

    """
    return data[to_list(columns)].copy()


def log_df_counts(
    data: pd.DataFrame,
    col: str,
    step_description: str,
    rows: bool = True,
    columns: bool = False,
) -> None:
    """Log num. of encounters and num. of samples (rows).

    Parameters
    ----------
    data: pandas.DataFrame
        Encounter specific input data.
    col: str
        Column name to count.
    step_description: str
        Description of intermediate processing step.
    rows: bool
        Log the number of samples, or rows of data.
    columns: bool
        Log the number of data columns.

    """
    LOGGER.info(step_description)
    num_encounters = data[col].nunique()
    if rows:
        num_samples = len(data)
        LOGGER.info("# samples: %d, # encounters: %d", num_samples, num_encounters)
    if columns:
        num_columns = len(data.columns)
        LOGGER.info("# columns: %d, # encounters: %d", num_columns, num_encounters)
