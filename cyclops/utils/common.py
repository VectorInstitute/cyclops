"""Common utility functions that can be used across multiple cyclops packages."""

import warnings
from datetime import datetime
from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.errors import PerformanceWarning


def to_timestamp(data: Union[pd.Series, npt.NDArray[Any]]) -> pd.Series:
    """Convert a Pandas series or NumPy array to a datetime/timestamp type.

    Parameters
    ----------
    data: pandas.Series or numpy.ndarray
        Data to be converted.

    Returns
    -------
    pandas.Series
        The converted data.

    """
    if isinstance(data, pd.Series):
        return pd.to_datetime(data)

    if isinstance(data, np.ndarray):
        return pd.to_datetime(pd.Series(data))

    raise ValueError(f"Type of data argument ({type(data)}) not supported.")


def add_years_approximate(
    timestamp_series: pd.Series,
    years_series: pd.Series,
) -> pd.Series:
    """Approximately add together a timestamp series with a years series row-by-row.

    Approximates are typically exact or incorrect by one day, e.g., on leap days.

    Parameters
    ----------
    timestamp_series: pandas.Series
        The series of timestamps to which to add.
    years_series: panadas.Series
        The series of years to add.

    Returns
    -------
    pandas.Series
        The timestamp series with the approximately added years.

    """
    # Add to the years column
    year = timestamp_series.dt.year + years_series

    # Handle the other columns
    month = timestamp_series.dt.month
    day = timestamp_series.dt.day
    hour = timestamp_series.dt.hour
    minute = timestamp_series.dt.minute

    # Create new timestamp column
    data = pd.DataFrame(
        {"year": year, "month": month, "day": day, "hour": hour, "minute": minute},
    )

    # Subtract 1 from potentially invalid leap days to avoid issues
    leap_days = (month == 2) & (day == 29)
    data["day"][leap_days] -= 1

    return pd.to_datetime(data)


def add_years_exact(timestamp_series: pd.Series, years_series: pd.Series) -> pd.Series:
    """Add together a timestamp series with a years series row-by-row.

    Warning: Very slow. It is worth using the add_years_approximate function even
    moderately large data.

    Parameters
    ----------
    timestamp_series: pandas.Series
        The series of timestamps to which to add.
    years_series: panadas.Series
        The series of years to add.

    Returns
    -------
    pandas.Series
        The timestamp series with the approximately added years.

    """
    warnings.warn(
        (
            "Computing the exact addition cannot be vectorized and is very slow. "
            "Consider using the quick, approximate calculation."
        ),
        PerformanceWarning,
        stacklevel=1,
    )
    return timestamp_series + years_series.apply(lambda x: pd.DateOffset(years=x))


def to_list(obj: Any) -> List[Any]:
    """Convert some object to a list of object(s) unless already one.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed object.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, (np.ndarray, set, dict)):
        return list(obj)

    return [obj]


def to_list_optional(
    obj: Optional[Any],
    none_to_empty: bool = False,
) -> Union[List[Any], None]:
    """Convert some object to a list of object(s) unless already None or a list.

    Parameters
    ----------
    obj : any
        The object to convert to a list.
    none_to_empty: bool, default = False
        If true, return a None obj as an empty list. Otherwise, return as None.

    Returns
    -------
    list or None
        The processed object.

    """
    if obj is None:
        if none_to_empty:
            return []
        return None

    return to_list(obj)


def to_datetime_format(date: str, fmt: str = "%Y-%m-%d") -> datetime:
    """Convert string date to datetime.

    Parameters
    ----------
    date: str
        Input date in string format.
    fmt: str, optional
        Date formatting string.

    Returns
    -------
    datetime
        Date in datetime format.

    """
    return datetime.strptime(date, fmt)


def list_swap(lst: List[Any], index1: int, index2: int) -> List[Any]:
    """Swap items in a list given the item index and new item index.

    Parameters
    ----------
    lst: list
        List in which elements will be swapped.
    index1: int
        Index of first item to swap.
    index2: int
        Index of second item to swap.

    Returns
    -------
    list
        List with elements swapped.

    """
    if not 0 <= index1 < len(lst):
        raise ValueError("index 1 is out of range.")

    if not 0 <= index2 < len(lst):
        raise ValueError("index 2 is out of range.")

    lst[index1], lst[index2] = lst[index2], lst[index1]

    return lst
