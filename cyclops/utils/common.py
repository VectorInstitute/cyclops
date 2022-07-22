"""Common utility functions that can be used across multiple cyclops packages."""

from datetime import datetime
from typing import Any, Union, List, Callable, Optional
from functools import wraps

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

    print(dict(list(dictionary.items())[0:10]))


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


def list_swap(lst: List, item1: int, item2: int) -> List:
    """Swap items in a list given the item index and new item index.

    Parameters
    ----------
    lst: list
        List in which elements will be swapped.
    item1: int
        Index of first item to swap.
    item2: int
        Index of second item to swap.

    Returns
    -------
    list
        List with elements swapped.

    """
    if not 0 <= item1 < len(lst):
        raise ValueError("Item1 index is out of range.")

    if not 0 <= item2 < len(lst):
        raise ValueError("Item2 index is out of range.")

    lst[item1], lst[item2] = lst[item2], lst[item1]

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
    to: str,
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
    if to == "array":
        in_fn = series_to_array
    elif to == "series":
        in_fn = array_to_series
    else:
        raise ValueError("to must be in: 'array', 'series'.")
    
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
        out_fn = lambda x: x
    else:
        raise ValueError(
            "out_to must be in: 'back', 'array', 'series', 'none'."
        )
    
    def decorator(func_: Callable) -> Callable:
        """Decorate function."""

        @wraps(func_)
        def wrapper_func(*args, **kwargs) -> Callable:
            # Convert relevant arguments.
            args = list(args)
            args, converted = zip(*[in_fn(arg) for arg in args])
            
            ret = func_(*tuple(args), **kwargs)
            
            if out_to == "back":
                if isinstance(ret, tuple):
                    if len(args) != len(ret):
                        raise ValueError(
                            ("When using out_to = 'back', the positional arguments "
                             "given must correspond to the values returned, i.e., "
                             "the same number in the same semantic ordering."
                            )
                        )
                    ret, _ = zip(*[
                        out_fn(r) if converted[i] else (r, False) for i, r in enumerate(ret)
                    ])
                    return tuple(ret)
                else:
                    if len(args) != 1:
                        raise ValueError(
                            ("When using out_to = 'back', the positional arguments "
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