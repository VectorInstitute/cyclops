import pandas as pd


def agg_mode(series: pd.Series) -> list:
    """
    Get the mode(s) of a series by using `.agg(agg_mode)`.

    Parameters
    ----------
    series : pd.Series
        Series.

    Returns
    -------
    list
        List containing the mode(s) of the input series.
    """
    return pd.Series.mode(series).to_list()


def groupby_agg_mode(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    single_modes_only: bool = False,
) -> pd.Series:
    """
    Compute the mode(s) for each group of a grouped series.

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
