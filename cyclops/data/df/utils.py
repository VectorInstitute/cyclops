"""Utility functions for working with Pandas DataFrames."""
from functools import reduce
from typing import (
    Any,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from cyclops.data.df.series_validation import is_bool_series
from cyclops.utils.common import to_list


COLS_TYPE = Union[Hashable, Sequence[Hashable]]


def check_cols(
    data: pd.DataFrame,
    cols: COLS_TYPE,
    raise_err_on_unexpected: bool = False,
    raise_err_on_existing: bool = False,
    raise_err_on_missing: bool = False,
) -> Tuple[Set[Hashable], Set[Hashable], Set[Hashable]]:
    """Check DataFrame columns for expected columns and handle errors.

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
    columns = set(to_list(cols))
    data_cols = set(data.columns)

    unexpected = data_cols - columns
    if raise_err_on_unexpected and len(unexpected) > 0:
        raise ValueError(f"Unexpected columns: {', '.join(unexpected)}")

    existing = data_cols.intersection(columns)
    if raise_err_on_existing and len(existing) > 0:
        raise ValueError(f"Existing columns: {', '.join(existing)}")

    missing = columns - data_cols
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
    """Combine non-overlapping DataFrames/Series into a single DataFrame/Series.

    The objects in `datas` should be all DataFrames or all Series, not a combination.

    For any given value location, it can be non-null in exactly 0 or 1 of the
    DataFrames. The combined DataFrame will contains all of these values.

    Parameters
    ----------
    datas : list of pandas.DataFrame or pandas.Series
        A list of DataFrames/Series to be combined.

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
        raise ValueError(
            "Unauthorized overlap found between DataFrames. Cannot combine.",
        )

    # Combine the DataFrames
    combined = datas[0].copy()
    for data in datas[1:]:
        combined = combined.combine_first(data)

    return combined


def reset_index_merge(
    left: Union[pd.DataFrame, pd.Series],
    right: Union[pd.DataFrame, pd.Series],
    index_col: Optional[COLS_TYPE] = None,
    **merge_kwargs: Any,
) -> pd.DataFrame:
    """Merge two dataframes after resetting their indexes.

    Parameters
    ----------
    left : pandas.DataFrame or pandas.Series
        The left object to merge.
    right : pandas.DataFrame or pandas.Series
        The right object to merge.
    index_col : hashable or sequence of hashable, optional
        Column(s) to set as index for the merged result.
    **merge_kwargs
        Additional keyword arguments to pass to pandas merge function.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    # Reset index for both dataframes
    left_reset = left.reset_index()
    right_reset = right.reset_index()

    # Merge the dataframes
    merged = pd.merge(left_reset, right_reset, **merge_kwargs)

    # If index_col is provided, set it for the merged dataframe
    if index_col:
        merged.set_index(index_col, inplace=True)

    return merged


def index_structure_equal(
    idx1: pd.Index,
    idx2: pd.Index,
    raise_err: bool = False,
) -> bool:
    """Check whether two indexes have the same structure.

    Values aren't considered.

    Parameters
    ----------
    idx1 : pandas.Index
        The first index to compare.
    idx2 : pandas.Index
        The second index to compare.
    raise_err : bool, default False
        If True, raises an error if indexes do not have the same structure.

    Returns
    -------
    bool
        True if the indexes have the same structure, otherwise False.
    """
    if type(idx1) != type(idx2):
        if raise_err:
            raise ValueError("Index dtypes do not match.")

        return False

    if idx1.names != idx2.names:
        if raise_err:
            raise ValueError("Index names do not match.")

        return False

    if idx1.nlevels != idx2.nlevels:
        if raise_err:
            raise ValueError("Number of index levels do not match.")

        return False

    return True


def is_multiindex(
    idx: pd.Index,
    raise_err: bool = False,
    raise_err_multi: bool = False,
) -> bool:
    """Check whether a given index is a MultiIndex.

    Parameters
    ----------
    idx : pd.Index
        Index to check.
    raise_err : bool, default False
        If True, raise a ValueError when idx is not a MultiIndex.
    raise_err_multi : bool, default False
        If True, raise a ValueError when idx is a MultiIndex.

    Raises
    ------
    ValueError
        Raised when `idx` is not a MultiIndex and `raise_err` is True.
        Raised when `idx` is a MultiIndex and `raise_err_multi` is True.

    Returns
    -------
    bool
        True if idx is a MultiIndex, False otherwise.
    """
    multiindex = isinstance(idx, pd.MultiIndex)

    if not multiindex and raise_err:
        raise ValueError("Index must be a MultiIndex.")

    if multiindex and raise_err_multi:
        raise ValueError("Index cannot be a MultiIndex.")

    return multiindex


def agg_mode(series: pd.Series) -> list[Any]:
    """Get the mode(s) of a series by using `.agg(agg_mode)`.

    Parameters
    ----------
    series : pd.Series
        Series.

    Returns
    -------
    list
        List containing the mode(s) of the input series.
    """
    return pd.Series.mode(series).to_list()  # type: ignore[no-any-return]


def groupby_agg_mode(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    single_modes_only: bool = False,
) -> pd.Series:
    """Compute the mode(s) for each group of a grouped series.

    Parameters
    ----------
    grouped : pd.core.groupby.generic.SeriesGroupBy
        Grouped series.
    single_modes_only : bool, default False
        If True, only groups with a singular mode are kept.

    Returns
    -------
    pd.Series
        A pandas Series containing the mode(s) for each group.
    """
    result = grouped.agg(agg_mode).explode()
    if single_modes_only:
        duplicate_indices = result.index[result.index.duplicated(keep=False)]
        result = result.drop(duplicate_indices)
    return result
