"""Functions for validating Pandas Series."""

from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)


def is_series(data: Any, raise_err: bool = False) -> bool:
    """Check if the input is a Pandas Series.

    Parameters
    ----------
    data : Any
        The input data to check.
    raise_err : bool, default False
        Whether to raise an error if the data is not a Series.

    Raises
    ------
    ValueError
        If `raise_err` is True and the input data is not a Pandas Series.

    Returns
    -------
    bool
        True if the input is a Pandas Series, False otherwise.
    """
    if isinstance(data, pd.Series):
        return True

    if raise_err:
        raise ValueError("Data must be a Pandas series.")

    return False


def is_bool_series(data: Any, raise_err: bool = False) -> bool:
    """Check if the input is a Pandas boolean series.

    Parameters
    ----------
    data : Any
        The input data to check.
    raise_err : bool, default False
        Whether to raise an error if the data is not a boolean Series.

    Raises
    ------
    ValueError
        If `raise_err` is True and the input data is not a Pandas boolean series.

    Returns
    -------
    bool
        True if the input is a Pandas boolean series, False otherwise.
    """
    if not is_series(data, raise_err=raise_err):
        return False

    if is_bool_dtype(data):
        return True

    if raise_err:
        raise ValueError("Pandas series must have a boolean type.")

    return False


def is_int_series(
    data: Any,
    raise_err: bool = False,
    raise_err_with_nullable: bool = False,
) -> bool:
    """Check if the input is a Pandas integer series.

    Parameters
    ----------
    data : Any
        The input data to check.
    raise_err : bool, default False
        Whether to raise an error if the data is not an integer Series.
    raise_err_with_nullable: bool, default False
        Whether to raise an error informing that, if the data is not an integer Series,
        consider a nullable integer data type. Takes precedence over raise_err.

    Raises
    ------
    ValueError
        If `raise_err` is True and the input data is not a Pandas integer series.

    Returns
    -------
    bool
        True if the input is a Pandas integer series, False otherwise.
    """
    if not is_series(data, raise_err=raise_err):
        return False

    if is_integer_dtype(data):
        return True

    if raise_err_with_nullable:
        raise ValueError(
            "Pandas series must have an integer type. Consider applying "
            "`series.astype('Int64')`, where Int64 is a nullable integer data type "
            "which enables the use of null values with an integer dtype.",
        )

    if raise_err:
        raise ValueError("Pandas series must have an integer type.")

    return False


def is_float_series(data: Any, raise_err: bool = False) -> bool:
    """Check if the input is a Pandas float series.

    Parameters
    ----------
    data : Any
        The input data to check.
    raise_err : bool, default False
        Whether to raise an error if the data is not a float Series.

    Raises
    ------
    ValueError
        If `raise_err` is True and the input data is not a Pandas float series.

    Returns
    -------
    bool
        True if the input is a Pandas float series, False otherwise.
    """
    if not is_series(data, raise_err=raise_err):
        return False

    if is_float_dtype(data):
        return True

    if raise_err:
        raise ValueError("Pandas series must have a float type.")

    return False


def is_str_series(data: Any, raise_err: bool = False) -> bool:
    """Check if the input is a Pandas string series.

    Parameters
    ----------
    data : Any
        The input data to check.
    raise_err : bool, default False
        Whether to raise an error if the data is not a string Series.

    Raises
    ------
    ValueError
        If `raise_err` is True and the input data is not a Pandas string series.

    Returns
    -------
    bool
        True if the input is a Pandas string series, False otherwise.
    """
    if not is_series(data, raise_err=raise_err):
        return False

    if is_string_dtype(data):
        return True

    if raise_err:
        raise ValueError("Pandas series must have a string type.")

    return False


def is_datetime_series(data: Any, raise_err: bool = False) -> bool:
    """Check if the input is a Pandas datetime series.

    Parameters
    ----------
    data : Any
        The input data to check.
    raise_err : bool, default False
        Whether to raise an error if the data is not a datetime Series.

    Raises
    ------
    ValueError
        If `raise_err` is True and the input data is not a Pandas datetime series.

    Returns
    -------
    bool
        True if the input is a Pandas datetime series, False otherwise.
    """
    if not is_series(data, raise_err=raise_err):
        return False

    if is_datetime64_any_dtype(data):
        return True

    if raise_err:
        raise ValueError("Pandas series must have a datetime type.")

    return False
