"""Aggregation functions."""

import logging
from collections import OrderedDict
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.cleaning import dropna_rows
from cyclops.processors.column_names import (
    RESTRICT_TIMESTAMP,
    START_TIMESTAMP,
    START_TIMESTEP,
    STOP_TIMESTAMP,
    TIMESTEP,
)
from cyclops.processors.constants import MEAN, MEDIAN
from cyclops.processors.impute import AggregatedImputer
from cyclops.processors.util import (
    fill_missing_timesteps,
    has_columns,
    is_timestamp_series,
)
from cyclops.utils.common import to_list, to_list_optional
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


AGGFUNCS = {MEAN: np.mean, MEDIAN: np.median}


class Aggregator:  # pylint: disable=too-many-instance-attributes
    """Equal-spaced aggregation, or binning, of temporal data.

    Computing aggregation metadata is expensive and should be done sparingly.

    Attributes
    ----------
    aggfuncs: dict
        Aggregation functions mapped from column to aggregation type.
        Each value is either function or string, e.g., {col_name: MEAN}.
        If a function, it should accept a series and return a single value.
    timestamp_col: str
        Name of the timestamp column in the data provided.
    time_by: list of str
        Name of columns by which to group to determine the bucket times.
    agg_by: list of str
        Name of columns by which to group to perform aggregation.
    timestep_size: float
        Time in hours for a single timestep, or bin.
    window_duration: float or None
        Time duration to consider after the start of a timestep.
    agg_meta_for: list of str or None
        Columns for which to compute aggregation metadata.
    window_times: pd.DataFrame or None
        The start/stop time windows used to aggregate the data.
    imputer: AggregatedImputer or None
        An imputer to perform aggregation.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        aggfuncs: Dict[str, Union[str, Callable]],
        timestamp_col: str,
        time_by: Union[str, List[str]],
        agg_by: Union[str, List[str]],
        timestep_size: int,
        window_duration: Optional[int] = None,
        imputer: Optional[AggregatedImputer] = None,
        agg_meta_for: Optional[List[str]] = None,
    ):
        """Init."""
        if agg_meta_for is not None:
            LOGGER.warning("Calculation of aggregation metadata slows aggregation.")

        self.aggfuncs = self._process_aggfuncs(aggfuncs)
        self.timestamp_col = timestamp_col
        self.time_by = to_list(time_by)
        self.agg_by = to_list(agg_by)
        self.timestep_size = timestep_size
        self.window_duration = window_duration
        self.agg_meta_for = to_list_optional(agg_meta_for)
        self.window_times = pd.DataFrame()  # Calculated when given the data
        self.imputer = imputer

        # Parameter checking
        if self.agg_meta_for is not None:
            if not set(self.agg_meta_for).issubset(set(list(self.aggfuncs))):
                raise ValueError(
                    "Cannot compute meta for a column not being aggregated."
                )

        if self.window_duration is not None:
            divided = self.window_duration / self.timestep_size
            if divided != int(divided):
                raise ValueError("Window duration be divisible by bucket size.")

    def get_timestamp_col(self) -> str:
        """Get timestamp column.

        Returns
        -------
        str
            Name of timestamp column.

        """
        return self.timestamp_col

    def get_aggfuncs(self) -> Dict[str, Callable]:
        """Get aggregation functions.

        Returns
        -------
        dict
            Aggregation functions.

        """
        return self.aggfuncs

    def _process_aggfuncs(
        self,
        aggfuncs: Dict[str, Union[str, Callable]],
    ) -> Dict[str, Any]:
        """Process aggregation functions for respective columns.

        Given a dict of values as functions or strings, convert a string to an
        aggfunc if recognized. Otherwise, simply return the functions.

        Returns
        -------
        dict
            The processed aggregation function dictionary.

        """
        for col, aggfunc in aggfuncs.items():
            if isinstance(aggfunc, str):
                if aggfunc not in AGGFUNCS:
                    raise ValueError(
                        f"""Aggfunc string {aggfunc} not supported.
                        Supporting: {','.join(AGGFUNCS)}"""
                    )
                aggfuncs[col] = AGGFUNCS[aggfunc]
            elif callable(aggfunc):
                pass
            else:
                raise ValueError("Aggfunc must be a string or callable.")

        return OrderedDict(aggfuncs)

    def _check_start_stop_window_ts(self, window_time: pd.DataFrame) -> None:
        """Check whether a window start/stop time have the correct format.

        Parameters
        ----------
        pandas.DataFrame
            Window start/stop time.

        """
        if not window_time.index.names == self.time_by:
            raise ValueError(f"Window start/stop times must have index: {self.time_by}")
        has_columns(
            window_time,
            [RESTRICT_TIMESTAMP],
            exactly=True,
            raise_error=True,
        )

    def _restrict_by_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:
        """Restrict events by the window start/stop times.

        Parameters
        ----------
        data: pandas.DataFrame
            Input data.

        Returns
        -------
        pandas.DataFrame
            The appropriately restricted data.

        """
        data = data.merge(self.window_times, on=self.time_by, how="left")

        cond = (data[self.timestamp_col] >= data[START_TIMESTAMP]) & (
            data[self.timestamp_col] <= data[STOP_TIMESTAMP]
        )

        # Keep if no match was made (i.e., no restriction performed)
        cond = cond | (data[self.timestamp_col].isnull())
        data = data[cond]
        return data

    def _use_provided_window(
        self, window_time: pd.DataFrame, default_time: pd.DataFrame, warning_args: Tuple
    ) -> pd.DataFrame:
        """Process a window start/stop time.

        Parameters
        ----------
        window_time: pandas.DataFrame
            The provided window start/stop time.
        default_time: pandas.DataFrame
            A default window start/stop time if the provided version is missing values.
        warning_args: tuple
            Tuple of strings used to format a warning message.

        Returns
        -------
            The processed provided window start/stop time.

        """
        self._check_start_stop_window_ts(window_time)
        index_missing = default_time.index.difference(window_time.index)
        if len(index_missing) > 0:
            LOGGER.warning(
                (
                    "Not all time_by groups have a specified window %s time. "
                    "Defaulting missing to %s time."
                ),
                *warning_args,
            )

            # Default non-existent to earliest time.
            window_time = default_time.join(window_time)
            inds = window_time[RESTRICT_TIMESTAMP].isna()
            window_time[RESTRICT_TIMESTAMP][inds] = window_time[self.timestamp_col][
                inds
            ]
            window_time = window_time.drop(self.timestamp_col, axis=1)
        return window_time

    def _compute_window_start(
        self,
        data: pd.DataFrame,
        window_start_time: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute the start timestamp for each time_by window.

        Parameters
        ----------
        data: pandas.DataFrame
            Data before aggregation.
        window_start_time: pd.DataFrame, optional
            An optionally provided window start time.

        Returns
        -------
        pandas.DataFrame
            Start timestamps for each time_by window.

        """
        # Take the earliest timestamp for each time_by group
        earliest_time = (
            data[self.time_by + [self.timestamp_col]]
            .groupby(self.time_by, sort=False)
            .agg({self.timestamp_col: "min"})
        )

        if window_start_time is None:
            # Use earliest times
            earliest_time = earliest_time.rename(
                {self.timestamp_col: RESTRICT_TIMESTAMP}, axis=1
            )
            window_start_time = earliest_time
        else:
            # Use provided start - with earliest times acting as default
            window_start_time = self._use_provided_window(
                window_start_time, earliest_time, ("start", "earliest")
            )

        self._check_start_stop_window_ts(window_start_time)
        return window_start_time

    def _compute_window_stop(
        self,
        data: pd.DataFrame,
        window_start_time: pd.DataFrame,
        window_stop_time: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute the stop timestamp for each time_by window.

        Parameters
        ----------
        data: pandas.DataFrame
            Data before aggregation.
        window_start_time: pd.DataFrame
            The window start time, which is necessary to compute the
            stop time when window_duration is set.
        window_stop_time: pd.DataFrame, optional
            An optionally provided window stop time.

        Returns
        -------
        pandas.DataFrame
            Stop timestamps for each time_by group.

        """
        # Use provided stop
        if window_stop_time is not None and self.window_duration is not None:
            raise ValueError(
                "Cannot provide window_stop_time if window_duration was set."
            )

        if self.window_duration is not None:
            # Use window duration to compute the stop times for each group
            window_stop_time = window_start_time.copy()
            window_stop_time[RESTRICT_TIMESTAMP] += pd.Timedelta(
                hours=self.window_duration
            )

        else:
            # Take the latest timestamp for each time_by group
            latest_time = (
                data[self.time_by + [self.timestamp_col]]
                .groupby(self.time_by, sort=False)
                .agg({self.timestamp_col: "max"})
            )

            if window_stop_time is None:
                # Use latest times
                latest_time = latest_time.rename(
                    {self.timestamp_col: RESTRICT_TIMESTAMP}, axis=1
                )
                window_stop_time = latest_time
            else:
                # Use provided stop - with latest times acting as default
                window_stop_time = self._use_provided_window(
                    window_stop_time, latest_time, ("stop", "latest")
                )

        self._check_start_stop_window_ts(window_stop_time)
        return window_stop_time

    def _compute_window_times(
        self,
        data: pd.DataFrame,
        window_start_time: Optional[pd.DataFrame] = None,
        window_stop_time: Optional[pd.DataFrame] = None,
    ):
        """Compute the start/stop timestamps for each time_by window.

        Parameters
        ----------
        data: pandas.DataFrame
            Data before aggregation.
        window_start_time: pd.DataFrame, optional
            An optionally provided window start time.
        window_stop_time: pd.DataFrame, optional
            An optionally provided window stop time.

        Returns
        -------
        pandas.DataFrame
            The start/stop timestamps for each time_by window.

        """
        # Compute window start time
        window_start_time = self._compute_window_start(
            data, window_start_time=window_start_time
        )

        # Compute window stop time
        window_stop_time = self._compute_window_stop(
            data, window_start_time, window_stop_time=window_stop_time
        )

        # Combine and compute additional information
        window_start_time = window_start_time.rename(
            {RESTRICT_TIMESTAMP: START_TIMESTAMP}, axis=1
        )
        window_stop_time = window_stop_time.rename(
            {RESTRICT_TIMESTAMP: STOP_TIMESTAMP}, axis=1
        )
        window_times = window_start_time.join(window_stop_time)

        return window_times

    def _compute_timestep(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute which timestep, or bin, each occurence falls into.

        Parameters
        ----------
        group: pandas.DataFrame
            A time_by group.

        Returns
        -------
        pandas.DataFrame
            The inputted group with an additional TIMESTEP column.

        """
        loc = tuple(group[self.time_by].values[0])
        start = self.window_times.loc[loc][START_TIMESTAMP]
        group[TIMESTEP] = (group[self.timestamp_col] - start) / pd.Timedelta(
            hours=self.timestep_size
        )
        group[TIMESTEP] = group[TIMESTEP].astype("int")

        return group

    def _compute_agg_meta(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute the aggregation metadata for an agg_by group.

        Parameters
        ----------
        group: pandas.DataFrame
            An agg_by group.

        Returns
        -------
        pandas.DataFrame
            The aggergation metadata information.

        """
        # Note: .counts() returns the number of non-null values in the Series.
        meta = group.agg(
            {col: [lambda x: x.count(), len] for col in self.agg_meta_for},
            dropna=False,
        )

        keep = []
        for col in self.agg_meta_for:
            meta[col + "_count"] = meta[(col, "len")]
            meta[col + "_null_fraction"] = 1 - (
                meta[(col, "<lambda_0>")] / meta[(col, "len")]
            )
            keep.extend([col + "_count", col + "_null_fraction"])

        meta = meta[keep]
        meta.columns = meta.columns.droplevel(1)
        return meta

    def _compute_aggregation(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute the aggregation for an agg_by group by timestep.

        Parameters
        ----------
        group: pandas.DataFrame
            An agg_by group.

        Returns
        -------
        pandas.DataFrame
            The aggregated group.

        """
        group = group.groupby(TIMESTEP, sort=False, dropna=False)

        # Compute aggregation meta
        if self.agg_meta_for is not None:
            agg_meta = self._compute_agg_meta(group)
        else:
            agg_meta = None

        if self.imputer is not None:
            if self.imputer.intra is not None:
                group = self.imputer.intra(group)

        AggregatedImputer(group)

        group = group.agg(self.aggfuncs)

        # Include aggregation meta
        if agg_meta is not None:
            group = group.join(agg_meta)

        return group

    def _aggregate(
        self, data: pd.DataFrame, include_timestep_start: bool = True
    ) -> pd.DataFrame:
        data_with_timesteps = data.groupby(self.time_by, sort=False).apply(
            self._compute_timestep
        )
        aggregated = data_with_timesteps.groupby(self.agg_by, sort=False).apply(
            self._compute_aggregation
        )

        if not include_timestep_start:
            return aggregated

        # Get the start timestamp for each timestep
        aggregated = aggregated.reset_index().set_index(self.time_by)

        aggregated = aggregated.join(self.window_times[START_TIMESTAMP])
        aggregated[START_TIMESTEP] = aggregated[START_TIMESTAMP] + pd.to_timedelta(
            aggregated[TIMESTEP] * self.timestep_size, unit="h"
        )
        aggregated = aggregated.drop(START_TIMESTAMP, axis=1)
        aggregated = aggregated.reset_index()

        if self.imputer is not None:
            if self.imputer.inter is not None:
                # aggregated = aggregated.set_index(self.agg_by)
                aggregated = aggregated.groupby(self.agg_by).apply(self.imputer.inter)
            # aggregated = aggregated.reset_index()

        aggregated = aggregated.set_index(self.agg_by + [TIMESTEP])

        return aggregated

    @time_function
    def __call__(
        self,
        data: pd.DataFrame,
        window_start_time: Optional[pd.DataFrame] = None,
        window_stop_time: Optional[pd.DataFrame] = None,
        include_timestep_start: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate.

        The window start and stop times can be used to cut short the timeseries.

        By default, the start time of a time_by group will be the earliest
        recorded timestamp in said group. Otherwise, a window_start_time
        can be provided by the user to override this default.

        The end time of a time_by group work similarly, but with the additional
        option of specifying a window_duration.

        Parameters
        ----------
        data: pandas.DataFrame
            Input data.
        window_start_time: pd.DataFrame, optional
            An optionally provided window start time.
        window_stop_time: pd.DataFrame, optional
            An optionally provided window stop time. This cannot be provided if
            window_duration was set.
        include_timestep_start: bool, default = True
            Whether to include the window start timestamps for each timestep.

        Returns
        -------
        pandas.DataFrame
            The aggregated data.

        """
        # Parameter checking
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data to aggregate must be a pandas.DataFrame.")

        has_columns(
            data,
            list(set([self.timestamp_col] + self.time_by + self.agg_by)),
            raise_error=True,
        )

        if has_columns(data, TIMESTEP):
            raise ValueError(f"Inputted data cannot have a column called {TIMESTEP}.")

        # Ensure the timestamp column is a timestamp. Drop null times (NaT).
        is_timestamp_series(data[self.timestamp_col], raise_error=True)
        data = dropna_rows(data, self.timestamp_col)

        # Compute start/stop timestamps
        self.window_times = self._compute_window_times(
            data, window_start_time=window_start_time, window_stop_time=window_stop_time
        )

        # Restrict the data according to the start/stop
        data = self._restrict_by_timestamp(data)

        return self._aggregate(data, include_timestep_start=include_timestep_start)

    def vectorize(self, aggregated: pd.DataFrame) -> np.ndarray:
        """Vectorize aggregated data.

        Parameters
        ----------
        aggregated: pandas.DataFrame
            Aggregated data.

        Returns
        -------
        numpy.ndarray
            Vectorized aggregated data of shape:
            (# of aggfuncs, *# of unique in each agg_by, window_duration/timestep_size)

        """
        if self.window_duration is None:
            raise NotImplementedError(
                "Cannot currently vectorize data aggregated with no window duration."
            )

        # Parameter checking
        has_columns(aggregated, list(self.aggfuncs.keys()), raise_error=True)
        if not aggregated.index.names == self.agg_by + [TIMESTEP]:
            raise ValueError(f"Index must be: {self.agg_by + [TIMESTEP]}.")

        def fill_agg_by_group_timesteps(group):
            group = fill_missing_timesteps(
                group, TIMESTEP, 0, self.window_duration / self.timestep_size
            )
            group = group.drop(TIMESTEP, axis=1)
            return group

        # Add missing agg_by groups
        aggregated = aggregated.reset_index()
        aggregated = aggregated[self.agg_by + [TIMESTEP] + list(self.aggfuncs.keys())]

        unique_individuals = [list(pd.unique(aggregated[col])) for col in self.agg_by]
        aggregated = aggregated.set_index(self.agg_by)
        existing_groups = set(aggregated.index.unique().values)
        all_groups = set(product(*unique_individuals))
        missing_groups = all_groups - existing_groups
        missing = pd.DataFrame(list(missing_groups), columns=self.agg_by)
        missing = missing.set_index(self.agg_by)
        aggregated = pd.concat([aggregated, missing])

        # Add missing timesteps
        vectorized = aggregated.groupby(self.agg_by).apply(fill_agg_by_group_timesteps)

        # ADD EXTRA IMPUTATION
        # if self.imputer is not None:
        #    if self.imputer.extra is not None:

        # Turn into numpy array and reshape accordingly
        unique_lens = [len(unique) for unique in unique_individuals]
        shape = unique_lens + [int(self.window_duration / self.timestep_size)]
        return np.stack(
            [vectorized[aggfunc].values.reshape(shape) for aggfunc in self.aggfuncs]
        ), list(self.aggfuncs.keys())
