"""Common utility functions that can be used across multiple cyclops packages."""

from datetime import datetime
from typing import Any, Optional

import numpy as np


def to_list(obj: Any) -> list:
    """Convert some object to a list of object(s) unless already one.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed function.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, np.ndarray):
        return list(obj)

    return [obj]


def to_list_optional(obj: Optional[Any]) -> Optional[list]:
    """Convert some object to a list of object(s) unless already None or a list.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed function.

    """
    if obj is None:
        return None

    if isinstance(obj, list):
        return obj

    if isinstance(obj, np.ndarray):
        return list(obj)

    return [obj]


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
