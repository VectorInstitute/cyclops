"""Aggregation functions."""

import logging
import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    DURATION,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    TIMESTEP,
    TIMESTEP_START_TIMESTAMP,
    WINDOW_START_TIMESTAMP,
    START_TIMESTAMP,
    START_TIMESTEP,
    STOP_TIMESTAMP,
)
from cyclops.processors.cleaning import dropna_rows
from cyclops.processors.constants import MEAN, MEDIAN, AGGFUNCS
from cyclops.processors.statics import compute_statics
from cyclops.processors.util import (
    assert_has_columns,
    gather_columns,
    has_columns,
    is_timestamp_series,
    log_counts_step,
)
from cyclops.utils.common import to_list, to_list_optional
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class Aggregator:
    """Equally-spaced aggregation, or bucketing, of temporal data.

    Parameters
    ----------
    aggfuncs: Dict
        Aggregation functions mapped from column to aggregation type.
        Each value is either function or string. If a string, e.g., ['mean', 'median'].
        If a function, it should accept a series and return a single value.
    bucket_size: float, optional
        Size of a single step in the time-series in hours.
        For example, if 2, temporal data is aggregated into bins of 2 hrs.
    window_duration: float, optional
        Window duration in hours, to consider for creating time-series.
        For example if its 100 hours, then all temporal data upto
        100 hours for a given encounter is considered. The start time of
        the time-series by default would be the earliest event recorded for that
        encounter, for example the first lab test. It can also be the admission
        time or a custom timestamp if provided.
    start_at_admission: bool, optional
        Flag to say if the window should start at the time of admission. This
        cannot be used in conjunction with 'start_at_admission'.
    start_window_ts: pandas.DataFrame
        Specific time for each encounter from which the window should start.
    stop_window_ts: pandas.DataFrame
        Specific time for each encounter upto which events should be considered.
        This cannot be used in conjunction with 'window'.
    agg_meta_for: list of str
        Columns for which to compute aggregation metadata.

    """
    def __init__(
        self,
        aggfuncs: Dict[str, Union[str, Callable]],
        timestamp_col: str,
        time_by: Union[str, List[str]],
        agg_by: Union[str, List[str]],
        bucket_size: int = 1,
        window_duration: Optional[int] = None,
        agg_meta_for: Optional[List[str]] = None,
    ):
        """Init."""
        if agg_meta_for is not None:
            LOGGER.warning(
                "Calculation of aggregation meta data significantly slows aggregation."
            )
        
        self.aggfuncs = self._process_aggfuncs(aggfuncs)
        self.timestamp_col = timestamp_col
        self.time_by = to_list(time_by)
        self.agg_by = to_list(agg_by)
        self.bucket_size = bucket_size
        self.window_duration = window_duration
        self.agg_meta_for = to_list_optional(agg_meta_for)
        
        if self.agg_meta_for is not None:
            if not set(self.agg_meta_for).issubset(set(list(self.aggfuncs))):
                raise ValueError(
                    "Cannot compute meta for a column not being aggregated."
                )
        
        if self.window_duration is not None:
            num = self.window_duration/self.bucket_size
            if num != int(num):
                LOGGER.warning(
                    "Suggested that the window duration be divisible by the bucket size."
                )
        
    
    def get_timestamp_col(self):
        return self.timestamp_col
    
    def get_aggfuncs(self):
        return self.aggfuncs
    
    def _process_aggfuncs(
        self,
        aggfuncs: Dict[str, Union[str, Callable]],
    ) -> Dict[str, Callable]:
        """Process aggregation functions for respective columns.

        Given a dict of values as functions or strings, convert a string to an aggfunc if
        recognized. Otherwise, simply return the functions.

        Returns
        -------
        Dict
            The processed aggregation functions.

        Raises
        ------
        NotImplementedError
            Asserts if supplied input strategy option is not recognized.

        """
        for col, aggfunc in aggfuncs.items():
            if isinstance(aggfunc, str):
                if aggfunc == MEAN:
                    aggfuncs[col] = np.mean
                elif aggfunc == MEDIAN:
                    aggfuncs[col] = np.median
                else:
                    raise NotImplementedError(
                        f"{aggfunc} is invalid. Supporting: {','.join(AGGFUNCS)}"
                    )
            elif callable(aggfunc):
                pass
            else:
                raise ValueError("Aggfunc must be a string or callable.")

        return aggfuncs
    
    def _check_start_stop_window_ts(self, window_ts) -> None:
        has_columns(
            window_ts,
            self.time_by + [self.timestamp_col],
            exactly=True,
            raise_error=True,
        )
    
    def _restrict_by_timestamp(self, data: pd.DataFrame):
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
        data = data.merge(self.window_times, on=self.time_by, how="left")

        cond = (data[self.timestamp_col] >= data[START_TIMESTAMP]) & \
            (data[self.timestamp_col] <= data[STOP_TIMESTAMP])

        # Keep if no match was made (i.e., no restriction performed)
        cond = cond | (data[self.timestamp_col].isnull())
        data = data[cond]
        return data


        return data

    def _compute_window_start(
        self,
        data: pd.DataFrame,
        window_start_time: Optional[pd.DataFrame] = None,
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
        relevant_cols = self.time_by + [self.timestamp_col]
        
        # Use provided start
        if window_start_time is not None:
            self._check_start_stop_window_ts(window_start_time)
        
        # Take the earliest timestamp for each time_by group
        else:
            has_columns(data, relevant_cols, raise_error=True)
            window_start_time = data[relevant_cols].groupby(self.time_by).agg({self.timestamp_col: "min"})
        
        return window_start_time


    def _compute_window_stop(
        self,
        data: pd.DataFrame,
        window_start_time: pd.DataFrame,
        window_stop_time: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute the end timestamp for each encounter window.

        If 'window' is provided, then end timestamp is computed as a fixed
        window length (hours) from the start timestamp ('start_time').
        If a custom dataframe with end timestamps for each encounter can be
        provided as the ending point of the window, that is used.

        Parameters
        ----------
        start_window_ts: pandas.DataFrame
            Start timestamp for each encounter window.

        Returns
        -------
        pandas.DataFrame
            Dataframe with end timestamps of windows for each encounter.

        """
        relevant_cols = self.time_by + [self.timestamp_col]
        
        # Use provided stop
        if window_stop_time is not None:
            self._check_start_stop_window_ts(window_stop_time)
            if self.window_duration is not None:
                raise ValueError("Cannot provide window_stop_time if window_duration was set.")
        
        # Use window duration to compute the stop times for each group
        elif self.window_duration is not None:
            window_stop_time = window_start_time.copy()
            window_stop_time[self.timestamp_col] += timedelta(hours=self.window_duration)
        
        # Take the latest timestamp for each time_by group
        else:
            relevant_cols = self.time_by + [self.timestamp_col]
            has_columns(data, relevant_cols, raise_error=True)
            window_stop_time = data[relevant_cols].groupby(self.time_by).agg({self.timestamp_col: "max"})
        
        return window_stop_time

    def _compute_window_times(
        self,
        data: pd.DataFrame,
        window_start_time: Optional[pd.DataFrame] = None,
        window_stop_time: Optional[pd.DataFrame] = None,
    ):
        # Compute window start time
        window_start_time = self._compute_window_start(
            data,
            window_start_time=window_start_time
        )
        
        # Compute window stop time
        window_stop_time = self._compute_window_stop(
            data,
            window_start_time,
            window_stop_time=window_stop_time
        )
        
        # Combine and compute additional information
        window_start_time = window_start_time.rename(
            {self.timestamp_col: START_TIMESTAMP}, axis=1
        )
        window_stop_time = window_stop_time.rename(
            {self.timestamp_col: STOP_TIMESTAMP}, axis=1
        )
        window_times = window_start_time.join(window_stop_time)
        window_times[DURATION] = (window_times[STOP_TIMESTAMP] - window_times[START_TIMESTAMP]) \
            / np.timedelta64(1, 'h')
        
        return window_times

    
    def _aggregate(self, data, with_timestep_start: bool = True):
        
        def compute_agg_meta(group):
            # Note: .counts() returns the number of non-null values in the Series.
            meta = group.agg({col: [lambda x: x.count(), len] for col in self.agg_meta_for}, dropna=False)
            
            keep = []
            for col in self.agg_meta_for:
                meta[col + "_count"] = meta[(col, "len")]
                meta[col + "_null_fraction"] = 1 - (meta[(col, "<lambda_0>")] / meta[(col, "len")])
                keep.extend([col + "_count", col + "_null_fraction"])
            
            meta = meta[keep]
            meta.columns = meta.columns.droplevel(1)
            return meta


        def compute_aggregation(group):
            group = group.groupby(TIMESTEP, dropna=False)
            
            # Compute aggregation meta
            if self.agg_meta_for != []:
                agg_meta = compute_agg_meta(group)
            else:
                agg_meta = None
            
            # TODO: Add intra imputation here.

            group = group.agg(self.aggfuncs)
            
            # Include aggregation meta
            if agg_meta is not None:
                group = group.join(agg_meta)
            
            return group

        def compute_timestep(group):
            loc = tuple(group[self.time_by].values[0])
            start = self.window_times.loc[loc][START_TIMESTAMP]
            group[TIMESTEP] = (
                group[self.timestamp_col] - start
            ) / pd.Timedelta(hours=self.bucket_size)
            group[TIMESTEP] = group[TIMESTEP].astype("int")
            
            return group
        
        
        data_with_timesteps = data.groupby(self.time_by).apply(compute_timestep)
        aggregated = data_with_timesteps.groupby(self.agg_by).apply(compute_aggregation)
        
        if not with_timestep_start:
            return aggregated
        
        # Compute the start timestamp for each bucket
        aggregated = aggregated.reset_index().set_index(self.time_by)
        aggregated = aggregated.join(self.window_times[START_TIMESTAMP])
        aggregated[START_TIMESTEP] = aggregated[START_TIMESTAMP] + pd.to_timedelta(
            aggregated[TIMESTEP] * self.bucket_size, unit="h"
        )
        aggregated = aggregated.drop(START_TIMESTAMP, axis=1)
        aggregated = aggregated.reset_index().set_index(self.agg_by + [TIMESTEP])
        return aggregated
    
    @time_function
    def __call__(
        self,
        data: pd.DataFrame,
        window_start_time: Optional[pd.DataFrame] = None,
        window_stop_time: Optional[pd.DataFrame] = None,
        with_timestep_start: bool = True,
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
        has_columns(
            data,
            list(set([self.timestamp_col] + self.time_by + self.agg_by)),
            raise_error=True
        )
        
        # Ensure the timestamp column is a timestamp. Drop null times (NaT).
        is_timestamp_series(data[self.timestamp_col], raise_error=True)
        data = dropna_rows(data, self.timestamp_col)
        
        # Compute start/stop timestamps 
        self.window_times = self._compute_window_times(
            data,
            window_start_time=window_start_time,
            window_stop_time=window_stop_time
        )
        
        # Restrict the data according to the start/stop
        data = self._restrict_by_timestamp(data)

        return self._aggregate(data)
