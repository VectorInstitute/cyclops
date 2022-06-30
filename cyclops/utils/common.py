"""Common utility functions that can be used across multiple cyclops packages."""

from datetime import datetime
from typing import Any, List, Optional

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
        The processed object.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, np.ndarray):
        return list(obj)

    return [obj]


def to_list_optional(obj: Optional[Any], none_to_empty: bool = False) -> Optional[list]:
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
