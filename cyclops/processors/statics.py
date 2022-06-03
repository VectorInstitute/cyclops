"""Processing functions for encounter based static data."""

import logging
from typing import Any, List, Union

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.processors.util import assert_has_columns, log_counts_step
from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@time_function
@assert_has_columns([ENCOUNTER_ID])
def infer_statics(
    data: pd.DataFrame, groupby_cols: Union[str, List[str]] = ENCOUNTER_ID
) -> List[str]:
    """Infer patient static columns using the unique values in a groupby object.

    Applies a groupby and counts unique values per column. If there is a single
    unique value (discounting NaNs) across all groups, then that column is considered
    to be a static feature column. By default, the encounter_id is used for groupby,
    although this function can be used more generally to perform groupby on other
    columns.

    Parameters
    ----------
    data: pandas.DataFrame
        Input DataFrame.
    groupby_cols: str or list of str
        Columns by which to group.

    Returns
    -------
    list of str:
        Names of the static columns.

    """
    groupby_cols = to_list(groupby_cols)
    if not set(groupby_cols).issubset(set(data.columns)):
        raise ValueError(f"{groupby_cols} must be a subset of {list(data.columns)}")

    grouped = data.groupby(groupby_cols)
    grouped_unique_limited = grouped.apply(lambda x: x.nunique(dropna=True) <= 1)
    num_one_unique = grouped_unique_limited.sum()
    static_cols = set(num_one_unique[num_one_unique == grouped.ngroups].index)
    LOGGER.info("Found %s static feature columns.", static_cols - {ENCOUNTER_ID})

    return list(static_cols)


@time_function
@assert_has_columns([ENCOUNTER_ID])
def compute_statics(
    data: pd.DataFrame, groupby_cols: Union[str, List[str]] = ENCOUNTER_ID
) -> List[str]:
    """Compute unique values from static columns (see infer_statics function).

    Parameters
    ----------
    data: pandas.DataFrame
        Input DataFrame.
    groupby_cols: str or list of str
        Columns by which to group.

    Returns
    -------
    pandas.DataFrame:
        Unique values of the static columns in each groupby object.

    """
    log_counts_step(data, "Computing static columns...", columns=True)
    static_cols = infer_statics(data, groupby_cols)
    statics = data[static_cols]
    grouped = statics.groupby(groupby_cols)

    def unique_non_null(series: pd.Series) -> Any:
        """Get a non-null unique value in the series, or if none, returns np.nan.

        Assumes at most one non-null unique value in the series.
        Parameters
        ----------
        series: pandas.Series
            Input Series.

        Returns
        -------
        any:
            Non-null unique value if it exists, otherwise np.nan

        """
        unique_vals = series.dropna().unique()
        return unique_vals[0] if len(unique_vals) == 1 else np.nan

    unique_statics = grouped.agg(unique_non_null)
    unique_statics = unique_statics.reset_index()
    unique_statics = unique_statics.set_index(groupby_cols)

    return unique_statics
