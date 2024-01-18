from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Sequence,
    Set,
    Tuple,
    Union,
)

from functools import reduce

import pandas as pd

from fecg.utils.common import to_list
from fecg.utils.pandas.type import (
    is_bool_series,
    is_int_series,
    is_series,
)

COLS_TYPE = Union[Hashable, Sequence[Hashable]]


def check_cols(
    data: pd.DataFrame,
    cols: COLS_TYPE,
    raise_err_on_unexpected: bool = False,
    raise_err_on_existing: bool = False,
    raise_err_on_missing: bool = False,
) -> Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]:
    """
    Check DataFrame columns for expected columns and handle errors.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to check columns against.
    cols : hashable or list of Hashable
        The column(s) to check for in the DataFrame.
    raise_err_on_unexpected : bool, default False
        Raise an error if unexpected columns are found.
    raise_err_on_existing : bool, default False
        Raise an error if any of the specified columns already exist.
    raise_err_on_missing : bool, default False
        Raise an error if any of the specified columns are missing.

    Returns
    -------
    Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]
        A tuple containing sets of unexpected, existing, and missing columns.
    """
    cols = set(to_list(cols))
    data_cols = set(data.columns)

    unexpected = data_cols - cols
    if raise_err_on_unexpected and len(unexpected) > 0:
        raise ValueError(f"Unexpected columns: {', '.join(unexpected)}")

    existing = data_cols.intersection(cols)
    if raise_err_on_existing and len(existing) > 0:
        raise ValueError(f"Existing columns: {', '.join(existing)}")

    missing = cols - data_cols
    if raise_err_on_missing and len(missing) > 0:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    return unexpected, existing, missing


def and_conditions(conditions: List[pd.Series]) -> pd.Series:
    """
    Perform element-wise logical AND operation on a list of boolean Series.

    Parameters
    ----------
    conditions : list of pd.Series
        A list of boolean Pandas Series.

    Raises
    ------
    ValueError
        If the conditions are not Pandas boolean series.

    Returns
    -------
    pd.Series
        A new Pandas Series resulting from the element-wise logical AND operation.
    """
    for condition in conditions:
        is_bool_series(condition, raise_err=True)

    return reduce(lambda x, y: x & y, conditions)


def or_conditions(conditions: List[pd.Series]) -> pd.Series:
    """
    Perform element-wise logical OR operation on a list of boolean Series.

    Parameters
    ----------
    conditions : list of pd.Series
        A list of boolean Pandas Series.

    Raises
    ------
    ValueError
        If the conditions are not Pandas boolean series.

    Returns
    -------
    pd.Series
        A new Pandas Series resulting from the element-wise logical OR operation.
    """
    for condition in conditions:
        is_bool_series(condition, raise_err=True)

    return reduce(lambda x, y: x | y, conditions)


def combine_nonoverlapping(datas: List[Union[pd.DataFrame, pd.Series]]) -> pd.DataFrame:
    """
    Combine non-overlapping DataFrames/Series into a single DataFrame/Series.

    The objects in `datas` should be all DataFrames or all Series, not a combination.

    For any given value location, it can be non-null in exactly 0 or 1 of the
    DataFrames. The combined DataFrame will contains all of these values.

    Parameters
    ----------
    datas : list of pandas.DataFrame
        A list of DataFrames to be combined.

    Returns
    -------
    pandas.DataFrame
        The combined DataFrame.

    Raises
    ------
    ValueError
        If unauthorized overlap is found between DataFrames.
    """
    # Get masks where the DataFrames are NaN
    datas_na = [data.isna() for data in datas]

    # Check that there is no unauthorized overlap
    datas_not_na = [(~data_na).astype(int) for data_na in datas_na]
    datas_not_na_sum = reduce(lambda x, y: x + y, datas_not_na)
    if not (datas_not_na_sum <= 1).all().all():
        raise ValueError("Unauthorized overlap found between DataFrames. Cannot combine.")

    # Combine the DataFrames
    combined = datas[0].copy()
    for data in datas[1:]:
        combined = combined.combine_first(data)

    return combined
