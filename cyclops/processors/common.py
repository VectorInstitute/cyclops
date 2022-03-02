"""Common re-useable processing functions."""

import pandas as pd

from cyclops.processors.column_names import ADMIT_TIMESTAMP


def filter_within_admission_window(
    data: pd.DataFrame, sample_ts_col_name, aggregation_window: int = 24
) -> pd.DataFrame:
    """Filter data based on single time window value.

    For e.g. if window is 24 hrs, then all data 24 hrs
    before time of admission and after 24 hrs of admission are considered.

    Parameters
    ----------
    data: pandas.DataFrame
        Data before filtering.
    sample_ts_col_name: str
        Name of column corresponding to the sample timestamp.
    aggregation_window: int, optional
        Window (no. of hrs) before and after admission to consider.

    Returns
    -------
    pandas.DataFrame
        Filtered data frame, with aggregates collected within window.
    """
    data_filtered = data.copy()
    sample_time = data_filtered[sample_ts_col_name]
    admit_time = data_filtered[ADMIT_TIMESTAMP]
    window_condition = abs((sample_time - admit_time) / pd.Timedelta(hours=1))
    data_filtered = data_filtered.loc[window_condition <= aggregation_window]
    return data_filtered
