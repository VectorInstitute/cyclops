"""Aggregation functions."""

import logging
import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    RESTRICT_TIMESTAMP,
    TIMESTEP,
    TIMESTEP_END_TIMESTAMP,
    TIMESTEP_START_TIMESTAMP,
    WINDOW_START_TIMESTAMP,
)
from cyclops.processors.constants import MEAN, MEDIAN
from cyclops.processors.statics import compute_statics
from cyclops.processors.util import (
    assert_has_columns,
    gather_columns,
    has_columns,
    is_timestamp_series,
    log_counts_step,
)
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@assert_has_columns([ENCOUNTER_ID, EVENT_TIMESTAMP])
def get_earliest_ts_encounter(timestamps: pd.DataFrame) -> pd.DataFrame:
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
    pandas.DataFrame
        A series with earliest timestamp among events for each encounter.

    """
    earliest_ts = timestamps.groupby(ENCOUNTER_ID).agg({EVENT_TIMESTAMP: "min"})[
        EVENT_TIMESTAMP
    ]
    earliest_ts = earliest_ts.reset_index()

    return earliest_ts


@assert_has_columns(
    [ENCOUNTER_ID, EVENT_TIMESTAMP],
    start=[ENCOUNTER_ID, RESTRICT_TIMESTAMP],
    stop=[ENCOUNTER_ID, RESTRICT_TIMESTAMP],
)
def restrict_events_by_timestamp(
    data: pd.DataFrame,
    start: Optional[pd.DataFrame] = None,
    stop: Optional[pd.DataFrame] = None,
):
    """Restrict events by the EVENT_TIMESTAMP.

    Restrict events by the EVENT_TIMESTAMP where, for a given ENCOUNTER_ID, events
    may be restricted to those only after the start timestamp and before the stop
    timestamp.

    The start/stop parameters are optionally specified depending on whether
    these timestamp restrictions are desired.

    If an ENCOUNTER_ID appears in data and not start/stop, then it will
    not have its events restricted. Every ENCOUNTER_ID in start/stop must appear
    in the data.

    If specified, the start and stop DataFrames expect columns ENCOUNTER_ID and
    RESTRICT_TIMESTAMP.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.
    start: pandas.DataFrame or None
        Restrict timestamps before the start time for a given ENCOUNTER_ID.
    stop: pandas.DataFrame or None
        Restrict timestamps after the stop time for a given ENCOUNTER_ID.

    Returns
    -------
    pandas.DataFrame
        The appropriately restricted data.

    """
    if start is None and stop is None:
        return data

    def restrict(data, restrict_df, is_start=True):
        # Assert correct columns.
        has_columns(restrict_df, [ENCOUNTER_ID, RESTRICT_TIMESTAMP], raise_error=True)

        # Assert that the encounter IDs in start/stop are a subset of those in data.
        assert restrict_df[ENCOUNTER_ID].isin(data[ENCOUNTER_ID]).all()

        # Assert that the time columns are the correct datatype.
        assert is_timestamp_series(restrict_df[RESTRICT_TIMESTAMP])

        data = data.merge(restrict_df, on=ENCOUNTER_ID, how="left")

        if is_start:
            cond = data[EVENT_TIMESTAMP] >= data[RESTRICT_TIMESTAMP]
        else:
            cond = data[EVENT_TIMESTAMP] <= data[RESTRICT_TIMESTAMP]

        # Keep if no match was made (i.e., no restriction performed)
        cond = cond | (data[RESTRICT_TIMESTAMP].isnull())
        data = data[cond]

        data.drop(columns=[RESTRICT_TIMESTAMP], inplace=True)
        return data

    if start is not None:
        data = restrict(data, start, is_start=True)
    if stop is not None:
        data = restrict(data, stop, is_start=False)

    return data


@dataclass
class Aggregator:
    """Aggregation of events to create bucketed temporal data (equally spaced).

    Parameters
    ----------
    aggfunc: Dict
        Aggregation function, either passed as function or string where if
        string, could be ['mean', 'median'].
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
        Flag to say if the window should start at the time of admission. This
        cannot be used in conjunction with 'start_at_admission'.
    start_window_ts: pandas.DataFrame, optional
        Specific time for each encounter from which the window should start.
    stop_window_ts: pandas.DataFrame, optional
        Specific time for each encounter upto which events should be considered.
        This cannot be used in conjunction with 'window'.

    """

    aggfunc: Dict[str, Union[str, Callable]] = field(default_factory=dict)
    bucket_size: int = 1
    window: int = 24
    start_at_admission: bool = False
    start_window_ts: Optional[pd.DataFrame] = None
    stop_window_ts: Optional[pd.DataFrame] = None
    meta: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Process aggregator arguments.

        Raises
        ------
        ValueError
            Incase user specifies both 'start_at_admission' and 'start_window_ts',
            an error is raised.

        """
        if self.start_at_admission and self.start_window_ts:
            raise ValueError("Can only have a unique starting point for window!")
        if self.stop_window_ts and self.window:
            LOGGER.error("window and stop_window_ts are both set, set window to None!")
            raise ValueError("Can only have unique ending point for window!")

        if isinstance(self.start_window_ts, pd.DataFrame):
            _ = has_columns(
                self.start_window_ts,
                [ENCOUNTER_ID, RESTRICT_TIMESTAMP],
                raise_error=True,
            )
        if isinstance(self.stop_window_ts, pd.DataFrame):
            _ = has_columns(
                self.stop_window_ts,
                [ENCOUNTER_ID, RESTRICT_TIMESTAMP],
                raise_error=True,
            )

        self.aggfunc = self._get_aggfunc()

    def _get_aggfunc(self) -> Dict[str, Union[str, Callable]]:
        """Get aggregation function(s) for respective columns.

        Given a dict of functions or strings, convert a string to an aggfunc if
        recognized. Otherwise, simply return functions.

        Returns
        -------
        Dict
            The aggregation functions for each column.

        Raises
        ------
        NotImplementedError
            Asserts if supplied input strategy option is not recognised (implemented).

        """
        if not bool(self.aggfunc):
            self.aggfunc = {EVENT_VALUE: np.mean}
        aggfunc_converted = {}
        for agg_col, aggfunc in self.aggfunc.items():
            if isinstance(aggfunc, str):
                if aggfunc == MEAN:
                    aggfunc_converted[agg_col] = np.mean
                elif aggfunc == MEDIAN:
                    aggfunc_converted[agg_col] = np.median
                else:
                    raise NotImplementedError(f"Provided {aggfunc} is not a valid one!")
            if callable(aggfunc):
                aggfunc_converted[agg_col] = aggfunc

        return aggfunc_converted

    @assert_has_columns([ENCOUNTER_ID])
    def compute_start_of_window(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the start timestamp for each encounter window.

        If 'start_at_admission' is True, then 'admit_timestamp' is returned,
        else if a custom dataframe with start timestamps for each encounter can be
        provided as the starting point of the window, that is used. Else by default,
        the start timestamp is computed as the earliest event timestamp for that
        encounter.

        Parameters
        ----------
        data: pandas.DataFrame
            Event data before aggregation.

        Returns
        -------
        pandas.DataFrame
            Dataframe with start timestamps of windows for each encounter.

        """
        if self.start_at_admission:
            start_time = gather_columns(data, [ENCOUNTER_ID, ADMIT_TIMESTAMP])
            start_time = compute_statics(start_time)
            start_time = start_time.reset_index()
            _ = has_columns(
                start_time, [ENCOUNTER_ID, ADMIT_TIMESTAMP], raise_error=True
            )
            start_time = start_time.rename(
                columns={ADMIT_TIMESTAMP: RESTRICT_TIMESTAMP}
            )

            return start_time

        if isinstance(self.start_window_ts, pd.DataFrame):
            start_time = self.start_window_ts

            return start_time

        start_time = get_earliest_ts_encounter(
            gather_columns(data, [ENCOUNTER_ID, EVENT_TIMESTAMP])
        )
        start_time = start_time.rename(columns={EVENT_TIMESTAMP: RESTRICT_TIMESTAMP})

        return start_time

    @assert_has_columns([ENCOUNTER_ID])
    def compute_end_of_window(
        self,
        start_time: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the end timestamp for each encounter window.

        If 'window' is provided, then end timestamp is computed as a fixed
        window length (hours) from the start timestamp ('start_time').
        If a custom dataframe with end timestamps for each encounter can be
        provided as the ending point of the window, that is used.

        Parameters
        ----------
        start_time: pandas.DataFrame
            Start timestamp for each encounter window.

        Returns
        -------
        pandas.DataFrame
            Dataframe with end timestamps of windows for each encounter.

        """
        if self.window:
            end_time = start_time.copy()
            _ = has_columns(
                end_time, [ENCOUNTER_ID, RESTRICT_TIMESTAMP], raise_error=True
            )
            end_time[RESTRICT_TIMESTAMP] += timedelta(hours=self.window)

        if isinstance(self.stop_window_ts, pd.DataFrame):
            end_time = self.stop_window_ts

        return end_time

    @assert_has_columns([ENCOUNTER_ID, WINDOW_START_TIMESTAMP])
    def aggregate_events(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate events into equally spaced buckets.

        All the event data is grouped based on encounters. First, the event data
        that lies within the window is restricted such that only those events are
        considered for aggregation. Given the aggreation options, the number of
        timesteps is determined, and the start of each timestep (timestamp) is added
        to the meta info. Event data columns are aggregated into the timestep bucket
        based on aggregation functions passed for each column.

        Parameters
        ----------
        data: pandas.DataFrame
            Input data.

        Returns
        -------
        tuple:
            tuple with Processed event features (pandas.DataFrame) and aggregation
            info like count of values in a bucket, fraction missing (pandas.DataFrame).

        """
        log_counts_step(data, "Aggregating event features...", columns=True)
        _ = has_columns(data, list(self.aggfunc.keys()), raise_error=True)

        def process_event(group):
            event_name = group[EVENT_NAME].iloc[0]
            group.drop(columns=[ENCOUNTER_ID, EVENT_NAME], axis=1)

            # To Fix: Figure out how to deal with missingness inside a bucket (intra).

            group = group.groupby(TIMESTEP, dropna=False)

            nonnull_count = group.agg({EVENT_VALUE: lambda x: x.count()}, dropna=False)
            nonnull_count.reset_index(inplace=True)
            nonnull_count.rename(columns={EVENT_VALUE: "nonnull_count"}, inplace=True)

            total_count = group.agg({EVENT_VALUE: len}, dropna=False)
            total_count.reset_index(inplace=True)
            total_count.rename(columns={EVENT_VALUE: "count"}, inplace=True)

            info = pd.merge(nonnull_count, total_count, how="inner", on=TIMESTEP)
            info["null_fraction"] = 1 - (info["nonnull_count"] / info["count"])
            info.drop(columns=["nonnull_count"], inplace=True)

            group = group.agg(self.aggfunc)
            group.reset_index(inplace=True)

            group[EVENT_NAME] = event_name
            group = pd.merge(group, info, how="left", on=TIMESTEP)

            return group

        def process_encounter(group):
            # Get timestep (bucket) for the timeseries events.
            group[TIMESTEP] = (
                group[EVENT_TIMESTAMP] - group[WINDOW_START_TIMESTAMP]
            ) / pd.Timedelta(hours=self.bucket_size)
            group[TIMESTEP] = group[TIMESTEP].astype("int")
            group.drop(EVENT_TIMESTAMP, axis=1, inplace=True)

            group = group.groupby([EVENT_NAME]).apply(process_event)
            group.reset_index(drop=True, inplace=True)

            return group

        # Group by encounters and process.
        grouped = data.groupby([ENCOUNTER_ID]).apply(process_encounter)
        grouped.reset_index(inplace=True)
        grouped.drop("level_1", axis=1, inplace=True)

        # Do we need this later?
        # grouped.dropna(inplace=True)

        return grouped

    def compute_timestep_timestamps(self, window_start_time: pd.DataFrame) -> tuple:
        """Compute the start and end timestamp for each timestep for each encounter.

        Parameters
        ----------
        window_start_time: pandas.DataFrame
            Dataframe with window start timestamps for each encounter.

        Returns
        -------
        tuple of pandas.DataFrame
            Dataframes with start and end timestamps for each timestep for each
            encounter.

        """
        timestep_start_times = window_start_time.copy()
        timesteps = pd.DataFrame(
            list(range(math.floor(self.window / self.bucket_size))), columns=[TIMESTEP]
        )
        timestep_start_times = timestep_start_times.merge(timesteps, how="cross")
        timestep_start_times[TIMESTEP_START_TIMESTAMP] = timestep_start_times[
            WINDOW_START_TIMESTAMP
        ] + pd.to_timedelta(timestep_start_times[TIMESTEP] * self.bucket_size, unit="h")
        timestep_start_times = timestep_start_times.set_index([ENCOUNTER_ID, TIMESTEP])
        timestep_start_times = timestep_start_times.drop(columns=WINDOW_START_TIMESTAMP)

        timestep_end_times = timestep_start_times + pd.to_timedelta(
            self.bucket_size, unit="h"
        )
        timestep_end_times = timestep_end_times.rename(
            columns={TIMESTEP_START_TIMESTAMP: TIMESTEP_END_TIMESTAMP}
        )

        return timestep_start_times, timestep_end_times

    @time_function
    @assert_has_columns([ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, EVENT_TIMESTAMP])
    def __call__(
        self,
        data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate events based on user provided parameters.

        Parameters
        ----------
        data: pandas.DataFrame
            Input data.

        Returns
        -------
        tuple:
            tuple with Processed event features (pandas.DataFrame) and aggregation
            info like count of values in a bucket, fraction missing (pandas.DataFrame).

        """
        window_start_time = self.compute_start_of_window(data)
        data = restrict_events_by_timestamp(data, start=window_start_time)
        window_end_time = self.compute_end_of_window(window_start_time)
        # Filter out those encounters with no events after window_start_time.
        window_end_time = window_end_time.loc[
            window_end_time[ENCOUNTER_ID].isin(data[ENCOUNTER_ID].unique())
        ]
        data = restrict_events_by_timestamp(data, stop=window_end_time)
        window_start_time = window_start_time.rename(
            columns={RESTRICT_TIMESTAMP: WINDOW_START_TIMESTAMP}
        )
        data = pd.merge(data, window_start_time, how="left", on=ENCOUNTER_ID)
        timestep_start_times, timestep_end_times = self.compute_timestep_timestamps(
            window_start_time
        )
        self.meta[TIMESTEP_START_TIMESTAMP] = timestep_start_times
        self.meta[TIMESTEP_END_TIMESTAMP] = timestep_end_times
        log_counts_step(data, "Restricting events within window...", columns=True)

        return self.aggregate_events(data)
