"""Aggregation functions."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    TIMESTEP,
)
from cyclops.processors.constants import MEAN, MEDIAN
from cyclops.processors.util import log_counts_step
from cyclops.query.util import to_list
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class Aggregator:
    """Aggregation options for temporal data.

    Parameters
    ----------
    strategy: str, optional
        Strategy to aggregate within bucket. ['mean', 'median']
    bucket_size: float, optional
        Size of a single step in the time-series in hours.
        For example, if 2, temporal data is aggregated into bins of 2 hrs.
    window: float, optional
        Window length in hours, to consider for creating time-series.
        For example if its 100 hours, then all temporal data upto
        100 hours for a given encounter is considered. The start time of
        the time-series by default would be the earliest event recorded for that
        encounter, for example the first lab test. It can also be the admission
        time or a custom timestamp if provided.
    start_at_admission: bool, optional
        Flag to say if the window should start at the time of admission.
    start_window_ts: datetime.datetime, optional
        Specific time from which the window should start.

    """

    aggfunc: Union[str, Callable] = MEAN
    bucket_size: int = 1
    window: int = 24
    start_at_admission: bool = False
    start_window_ts: Optional[datetime] = None

    def __post_init__(self):
        """Process aggregator arguments."""
        self.aggfunc = self._get_aggfunc()

    def _get_aggfunc(self) -> Callable:
        """Return an aggregation function.

        Given a function or string, convert a string to an aggfunc if
        recognized. Otherwise, simply return the same Callable object.

        Parameters
        ----------
        strategy: str, optional
            Strategy for aggregation, options are ['mean', 'median']

        Returns
        -------
        Callable
            The aggregation function.

        Raises
        ------
        NotImplementedError
            Asserts if supplied input strategy option is not recognised (implemented).

        """
        if isinstance(self.aggfunc, str):
            if self.aggfunc == MEAN:
                return np.mean
            if self.aggfunc == MEDIAN:
                return np.median

            raise NotImplementedError(f"Provided {self.aggfunc} is not a valid one!")

        return self.aggfunc


def get_earliest_ts_encounter(timestamps: pd.DataFrame) -> pd.Series:
    """Get the timestamp of the earliest event for an encounter.

    Given 2 columns, i.e. encounter ID, and timestamp of events,
    this function finds the timestamp of the earliest event for an
    encounter.

    Parameters
    ----------
    timestamps: pandas.DataFrame
        Event timestamps and encounter IDs in a dataframe.

    Returns
    -------
    pandas.Series
        A series with earliest timestamp among events for each encounter.

    """
    earliest_ts = timestamps.groupby(ENCOUNTER_ID).agg({EVENT_TIMESTAMP: "min"})[
        EVENT_TIMESTAMP
    ]

    return earliest_ts


@time_function
def filter_upto_window(
    data: pd.DataFrame,
    window: int = 24,
    start_at_admission: bool = False,
    start_window_ts: datetime = None,
) -> pd.DataFrame:
    """Filter data based on window value.

    For e.g. if window is 24 hrs, then all data for the encounter
    upto after 24 hrs after the first event timestamp are considered. If
    'start_at_admission' is True, the all events before admission are dropped.
    Optionally, a start timestamp can be provided as the starting point to the
    window.

    Parameters
    ----------
    data: pandas.DataFrame
        Data before filtering.
    window: int, optional
        Window (no. of hrs) upto after admission to consider.
    start_at_admission: bool, optional
        Flag to say if the window should start at the time of admission.
    start_window_ts: datetime.datetime, optional
        Specific time from which the window should start.

    Returns
    -------
    pandas.DataFrame
        Filtered data frame, with aggregates collected within window.

    Raises
    ------
    ValueError
        Incase user specifies both 'start_at_admission' and 'start_window_ts',
        an error is raised.

    """
    if start_at_admission and start_window_ts:
        raise ValueError("Can only have a unique starting point for window!")

    data_filtered = data.copy()
    sample_time = data_filtered[EVENT_TIMESTAMP]

    if start_at_admission:
        start_time = data_filtered[ADMIT_TIMESTAMP]
    if start_window_ts:
        start_time = start_window_ts
    else:
        start_time = get_earliest_ts_encounter(
            data_filtered[[ENCOUNTER_ID, EVENT_TIMESTAMP]]
        )
        start_time = start_time[data_filtered[ENCOUNTER_ID]].reset_index()
        start_time = start_time[EVENT_TIMESTAMP]
        start_time.index = sample_time.index

    data_filtered = data_filtered.loc[sample_time >= start_time]
    window_condition = (sample_time - start_time) / pd.Timedelta(hours=1)
    data_filtered = data_filtered.loc[window_condition <= window]

    return data_filtered


def gather_events_into_single_bucket(
    data: pd.DataFrame, aggregator: Aggregator
) -> pd.DataFrame:
    """Gather events into single bucket.

    If aggregation window and bucket size are the same, then
    all events fall into the same bucket, and hence instead of a
    time-series, a single feature value per event is gathered.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.
    aggregation_strategy: str
        Aggregation strategy within bucket.

    Returns
    -------
    pandas.DataFrame:
        Processed event features.

    """
    features = pd.pivot_table(
        data,
        values=EVENT_VALUE,
        index=ENCOUNTER_ID,
        columns=[EVENT_NAME],
        aggfunc=aggregator.aggfunc,
        dropna=False,
    )

    return features, None


@time_function
def gather_event_features(data: pd.DataFrame, aggregator: Aggregator) -> pd.DataFrame:
    """Gather events from encounters into time-series features.

    All the event data is grouped based on encounters. For each
    encounter, the number of timesteps is determined, and the
    event value for each event belonging to a timestep
    is gathered accordingly to create a DataFrame of features,
    where the number of feature columns is equal to number of
    event names, e.g. lab tests + vital measurements. The features
    DataFrame is then indexable using encounter_id and timestep.
    If 'aggregator.window' and 'aggregator.bucket_size' are the same,
    a faster groupby, since the result is a single feature value (i.e. single
    timestep) and not a time-series.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.
    aggregator: cyclops.processor.Aggregator
        Aggregation options.

    Returns
    -------
    pandas.DataFrame:
        Processed event features.
    pandas.DataFrame:
        Aggregation information.

    """
    log_counts_step(data, "Gathering event features...", columns=True)

    data = filter_upto_window(
        data,
        window=aggregator.window,
        start_at_admission=aggregator.start_at_admission,
        start_window_ts=aggregator.start_window_ts,
    )
    log_counts_step(data, "Filtering events within window...", columns=True)

    # All events are placed in a single bucket, hence not a time-series.
    if aggregator.window == aggregator.bucket_size:
        return gather_events_into_single_bucket(data, aggregator)

    num_timesteps = math.floor(aggregator.window / aggregator.bucket_size)

    def fill_missing_range(data, col, range_from, range_to, fill_with=np.nan):
        return (
            data.merge(
                how="right",
                on=col,
                right=pd.DataFrame({col: np.arange(range_from, range_to)}),
            )
            .sort_values(by=col)
            .reset_index()
            .fillna(fill_with)
            .drop(["index"], axis=1)
        )

    def process_event(group):
        event_name = group[EVENT_NAME].iloc[0]
        group.drop(columns=[ENCOUNTER_ID, EVENT_NAME], axis=1)

        # ADD IMPUTATION METHOD

        group = group.groupby(TIMESTEP, dropna=False)

        nonnull_count = group.agg({EVENT_VALUE: lambda x: x.count()}, dropna=False)
        nonnull_count.reset_index(inplace=True)
        nonnull_count.rename(columns={EVENT_VALUE: "nonnull_count"}, inplace=True)

        total_count = group.agg({EVENT_VALUE: lambda x: len(x)}, dropna=False)
        total_count.reset_index(inplace=True)
        total_count.rename(columns={EVENT_VALUE: "count"}, inplace=True)

        info = pd.merge(nonnull_count, total_count, how="inner", on=TIMESTEP)
        info["null_fraction"] = 1 - (info["nonnull_count"] / info["count"])
        info.drop(columns=["nonnull_count"], inplace=True)

        group = group.agg({EVENT_VALUE: aggregator.aggfunc})
        group.reset_index(inplace=True)

        group = fill_missing_range(group, TIMESTEP, 0, num_timesteps)

        group[EVENT_NAME] = event_name

        group = pd.merge(group, info, how="left", on=TIMESTEP)

        return group

    def process_encounter(group):
        # Get timestep (bucket) for the timeseries events
        group[TIMESTEP] = (
            group[EVENT_TIMESTAMP] - min(group[EVENT_TIMESTAMP])
        ) / pd.Timedelta(hours=aggregator.bucket_size)
        group[TIMESTEP] = group[TIMESTEP].astype("int")
        group.drop(EVENT_TIMESTAMP, axis=1, inplace=True)

        group = group.groupby([EVENT_NAME]).apply(process_event)
        group.reset_index(drop=True, inplace=True)

        return group

    # Drop unwanted columns
    data.drop(ADMIT_TIMESTAMP, axis=1, inplace=True)

    # Group by encounters and process
    grouped = data.groupby([ENCOUNTER_ID]).apply(process_encounter)
    grouped.reset_index(inplace=True)
    grouped.drop("level_1", axis=1, inplace=True)

    features = pd.pivot_table(
        grouped.drop(columns=["count", "null_fraction"]),
        values=EVENT_VALUE,
        index=[ENCOUNTER_ID, TIMESTEP],
        columns=[EVENT_NAME],
        aggfunc=aggregator.aggfunc,
        dropna=False,
    )

    grouped.dropna(inplace=True)

    return features, grouped


@time_function
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
def gather_statics(
    data: pd.DataFrame, groupby_cols: Union[str, List[str]] = ENCOUNTER_ID
) -> List[str]:
    """Gather unique values from static columns (see infer_statics function).

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
    log_counts_step(data, "Gathering static features...", columns=True)
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
