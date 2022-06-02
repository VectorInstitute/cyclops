"""Aggregation functions."""

import logging
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

    If specified, the start and stop DataFrames expect columns ENCOUNTER_ID, "time".

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
        # Assert correct columns
        has_columns(restrict_df, [ENCOUNTER_ID, RESTRICT_TIMESTAMP], raise_error=True)
        # Assert that the encounter IDs in start/stop are a subset of those in data
        assert restrict_df[ENCOUNTER_ID].isin(data[ENCOUNTER_ID]).all()

        # Assert that the time columns are the correct datatype
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
    aggfunc: str or Callable, optional
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

    aggfunc: Union[str, Callable] = MEAN
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

            group = group.agg({EVENT_VALUE: self.aggfunc})
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
        start_time = self.compute_start_of_window(data)
        end_time = self.compute_end_of_window(start_time)
        data = restrict_events_by_timestamp(data, start_time, end_time)
        start_time = start_time.rename(
            columns={RESTRICT_TIMESTAMP: WINDOW_START_TIMESTAMP}
        )
        data = pd.merge(data, start_time, how="left", on=ENCOUNTER_ID)
        log_counts_step(data, "Restricting events within window...", columns=True)

        return self.aggregate_events(data)
