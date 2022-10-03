"""Common utility functions that can be used across multiple cyclops packages."""

import warnings
from datetime import datetime
from functools import wraps
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


def add_years_approximate(
    timestamp_series: pd.Series, years_series: pd.Series
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
        {"year": year, "month": month, "day": day, "hour": hour, "minute": minute}
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
    )
    return timestamp_series + years_series.apply(lambda x: pd.DateOffset(years=x))


def to_list(obj: Any) -> list:
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
    obj: Optional[Any], none_to_empty: bool = False
) -> Union[list, None]:
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


def print_dict(dictionary: dict, limit: int = None) -> None:
    """Print a dictionary with the option to limit the number of items.

    Parameters
    ----------
    dictionary: dict
        Dictionary to print.
    limit: int, optional
        Item limit to print.

    """
    if limit is None:
        print(dictionary)
        return

    if limit < 0:
        raise ValueError("Limit must be greater than 0.")

    print(dict(list(dictionary.items())[0:limit]))


def append_if_missing(lst: Any, append_lst: Any, to_start: bool = False) -> List[Any]:
    """Append objects in append_lst to lst if not already there.

    Parameters
    ----------
    lst: any
        An object or list of objects.
    append_lst: any
        An object or list of objects to append.
    to_start: bool
        Whether to append the objects to the start or end.

    Returns
    -------
    list of any
        The appended list.

    """
    lst = to_list(lst)
    append_lst = to_list(append_lst)
    extend_lst = [col for col in append_lst if col not in lst]

    if to_start:
        return extend_lst + lst

    return lst + extend_lst


def to_datetime_format(date: str, fmt="%Y-%m-%d") -> datetime:
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


def list_swap(lst: List, index1: int, index2: int) -> List:
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


def is_one_dimensional(arr: np.ndarray, raise_error: bool = True):
    """Determine whether a NumPy array is 1-dimensional.

    Parameters
    ----------
    arr: numpy.ndarray
        Array to check.
    raise_error: bool, default = True
        Whether to raise an error if the array is not 1-dimensional.

    Returns
    -------
    bool
        Whether the NumPy array is 1-dimensional.

    """
    if arr.ndim == 1:
        return True

    if raise_error:
        raise ValueError("Array must be one dimensional.")

    return False


def series_to_array(val: Any) -> Any:
    """Convert Pandas series to NumPy array, leaving other values unchanged.

    Parameters
    ----------
    val: any
        Any value.

    Returns
    -------
    any
        Return a NumPy array if a Pandas series is given, otherwise return the
        value unchanged.

    """
    if isinstance(val, pd.Series):
        return val.values, True
    return val, False


def array_to_series(val: Any):
    """Convert NumPy array to Pandas series, leaving other values unchanged.

    Parameters
    ----------
    val: any
        Any value.

    Returns
    -------
    any
        Return a Pandas series if a NumPy array is given, otherwise return the
        value unchanged.

    """
    if isinstance(val, np.ndarray):
        is_one_dimensional(val)
        return pd.Series(data=val), True
    return val, False


def array_series_conversion(
    to: str,  # pylint: disable=invalid-name
    out_to: str = "back",
) -> Callable:
    """Convert positional arguments between numpy.ndarray and pandas.Series.

    When using out_to = 'back', the positional arguments given must correspond to the
    values returned, i.e., the same number in the same semantic ordering.

    Parameters
    ----------
    to: str
        The type to which to convert positional arguments. Options are
        'array' or 'series'.
    out_to: str
        The type to which to convert return types. Options are
        'back', 'array', 'series', or 'none'. 'back' will convert the
        returned values to the same as was inputted (the positional arguments
        given must correspond to the values returned, i.e., the same number in
        the same semantic ordering.

    Returns
    -------
    callable
        The processed function.

    """
    in_fn: Callable
    if to == "array":
        in_fn = series_to_array
    elif to == "series":
        in_fn = array_to_series
    else:
        raise ValueError("to must be in: 'array', 'series'.")

    def identity(val: Any):
        return val, False

    out_fn: Callable
    if out_to == "back":
        if to == "array":
            out_fn = array_to_series
        elif to == "series":
            out_fn = series_to_array
    elif out_to == "array":
        out_fn = series_to_array
    elif out_to == "series":
        out_fn = array_to_series
    elif out_to == "none":
        out_fn = identity
    else:
        raise ValueError("out_to must be in: 'back', 'array', 'series', 'none'.")

    def decorator(func_: Callable) -> Callable:
        """Decorate function."""

        @wraps(func_)
        def wrapper_func(*args, **kwargs):
            # Convert relevant arguments.
            args = list(args)
            args, converted = zip(*[in_fn(arg) for arg in args])

            ret = func_(*tuple(args), **kwargs)
            print(len(args), len(ret), out_to, type(ret))
            print("----")

            if out_to == "back":
                # Multiple returns
                if isinstance(ret, tuple):
                    if len(args) != len(ret):
                        raise ValueError(
                            (
                                "When using out_to = 'back', the positional arguments "
                                "given must correspond to the values returned, i.e., "
                                "the same number in the same semantic ordering."
                            )
                        )
                    ret, _ = zip(
                        *[
                            out_fn(r) if converted[i] else (r, False)
                            for i, r in enumerate(ret)
                        ]
                    )
                    return tuple(ret)

                # One return
                if len(args) != 1:
                    raise ValueError(
                        (
                            "When using out_to = 'back', the positional arguments "
                            "given must correspond to the values returned, i.e., "
                            "the same number in the same semantic ordering."
                        )
                    )
                if converted[0]:
                    ret, _ = out_fn(ret)
                    return ret

                return ret

            if isinstance(ret, tuple):
                ret, _ = zip(*[out_fn(r) for r in ret])
                return tuple(ret)

            ret, _ = out_fn(ret)
            return ret

        return wrapper_func

    return decorator
