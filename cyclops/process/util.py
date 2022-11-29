"""Utility functions for processor API."""

import logging
from functools import wraps
from typing import Callable, List, Optional, Union

import pandas as pd

from cyclops.process.column_names import ENCOUNTER_ID
from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def create_indicator_variables(
    features: pd.DataFrame, columns: Optional[List] = None
) -> pd.DataFrame:
    """Create binary indicator variable for each column (or specified).

    Create new indicator variable columns based on NAs for other feature columns.

    Parameters
    ----------
    features: pandas.DataFrame
        Input features with missing values.
    columns: list, optional
        Columns to create variables, all if not specified.

    Returns
    -------
    pandas.DataFrame
        Dataframe with indicator variables as columns.

    """
    if columns:
        indicator_features = features[columns]
    else:
        indicator_features = features

    return indicator_features.notnull().astype(int).add_suffix("_indicator")


def is_timestamp_series(series, raise_error: bool = False):
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
            dataframe_args = [i for i in fn_args if isinstance(i, pd.DataFrame)]

            assert len(args) <= len(dataframe_args)

            for i, arg in enumerate(args):
                if arg is None:  # Can specify None to skip over checking a DataFrame
                    continue
                has_columns(dataframe_args[i], arg, raise_error=True)

            for key, required_cols in kwargs.items():
                # If an optional DataFrame is not provided, it will be skipped
                if key not in fn_kwargs:
                    continue

                assert isinstance(fn_kwargs[key], pd.DataFrame)
                has_columns(fn_kwargs[key], required_cols, raise_error=True)

            return func_(*fn_args, **fn_kwargs)

        return wrapper_func

    return decorator


def has_range_index(data: pd.DataFrame) -> bool:
    """Check whether a DataFrame has a range index.

    Parameters
    ----------
    data: pandas.DataFrame
        Data.

    Returns
    -------
    bool
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
