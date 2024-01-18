import pandas as pd


def index_structure_equal(
    idx1: pd.Index,
    idx2: pd.Index,
    raise_err: bool = False,
) -> bool:
    """
    Check whether two indexes have the same structure. Values aren't considered.

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
    """
    Check whether a given index is a MultiIndex.

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
