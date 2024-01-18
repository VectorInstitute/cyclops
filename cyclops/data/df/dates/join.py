from typing import Hashable, Optional, Sequence, Union

import pandas as pd

from fecg.utils.pandas.pandas import COLS_TYPE


def reset_index_merge(
    left: Union[pd.DataFrame, pd.Series],
    right: Union[pd.DataFrame, pd.Series],
    index_col: Optional[COLS_TYPE] = None,
    **merge_kwargs,
) -> pd.DataFrame:
    """
    Merges two dataframes after resetting their indexes.

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
