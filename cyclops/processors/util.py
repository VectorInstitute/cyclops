"""Utility functions for processor API."""

import logging
from functools import wraps
from typing import Callable

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


def is_timestamp_series(series):
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
    return series.dtype == pd.to_datetime(["2069-03-29 02:30:00"]).dtype


def has_columns(
    data: pd.DataFrame, required_columns: list, raise_error: bool = False
) -> bool:
    """Check if data has required columns for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    required_columns: list
        List of column names that must be present in data.
    raise_error: bool
        Whether to raise a ValueError if there are missing columns.

    Returns
    -------
    bool
        True if all required columns are present, otherwise False.

    """
    required_set = set(required_columns)
    columns = set(data.columns)
    present = required_set.issubset(columns)

    if not present and raise_error:
        missing = required_set - columns
        raise ValueError(f"Missing required columns {missing}")

    return present


def assert_has_columns(*args, **kwargs) -> Callable:
    """Decorate function to assert that input DataFrames have the necessary columns.

    assert_has_columns(["A", "B"], None) is equivalent to assert_has_columns(["A", "B"])
    but may be necessary when wanting to check,
    assert_has_columns(["A"], None, ["C"])

    Can also check keyword arguments, e.g., optional DataFrames,
    assert_has_columns(["A"], optional_df=["D"])

    Parameters
    ----------
    *args
        Required columns of the function's ordered DataFrame arguments.
    **kwargs
        Keyword corresponds to the DataFrame kwargs of the function.
        The value is this keyword argument's required columns.

    Returns
    -------
    Callable
        Decorator function.

    """

    def decorator(func_: Callable) -> Callable:
        @wraps(func_)
        def wrapper_func(*fn_args, **fn_kwargs) -> Callable:
            # Check only the DataFrame arguments
            dataframe_fn_args = [i for i in fn_args if isinstance(i, pd.DataFrame)]

            assert len(args) <= len(dataframe_fn_args)

            for i, arg in enumerate(args):
                if arg is None:  # Can specify None to skip over checking a DataFrame
                    continue
                has_columns(dataframe_fn_args[i], arg, raise_error=True)

            for key, required_cols in kwargs.items():
                # If an optional DataFrame is not provided, it will be skipped
                if key not in fn_kwargs:
                    continue

                assert isinstance(fn_kwargs[key], pd.DataFrame)
                has_columns(fn_kwargs[key], required_cols, raise_error=True)

            return func_(*fn_args, **fn_kwargs)

        return wrapper_func

    return decorator


def gather_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Gather specified columns, discarding rest and return copy of columns.

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
    return data[columns].copy()


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
