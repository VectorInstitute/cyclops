"""Aggregation functions."""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    TIMESTEP,
)
from cyclops.processors.utils import log_counts_step
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
    strategy: str
        Strategy to aggregate within bucket. ['mean', 'median']
    bucket_size: float
        Size of a single step in the time-series in hours.
        For example, if 2, temporal data is aggregated into bins of 2 hrs.
    window: float
        Window length in hours, to consider for creating time-series.
        For example if its 100 hours, then all temporal data upto
        100 hours after admission time, for a given encounter is
        considered. This can be negative as well, in which case,
        events from the patient's time in the ER will be considered.

    """

    strategy: str = "mean"
    bucket_size: int = 1
    window: int = 24


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
    start_time = min(sample_time)
    if start_at_admission:
        start_time = data_filtered[ADMIT_TIMESTAMP]
    if start_window_ts:
        start_time = start_window_ts
    data_filtered = data_filtered.loc[sample_time >= start_time]
    window_condition = (sample_time - start_time) / pd.Timedelta(hours=1)
    data_filtered = data_filtered.loc[window_condition <= window]
    return data_filtered


def aggregate_values_in_bucket(values: pd.Series, strategy: str = "mean") -> np.float64:
    """Aggregate multiple values within a bucket into single value.

    Based on the strategy, collapse multiple values into a single
    value. For example, if strategy is 'mean', then return
    mean of values.

    Parameters
    ----------
    values: pandas.Series
        List of input values that fall within the same bucket.
    strategy: str, optional
        Strategy for aggregation, options are ['mean', 'median']

    Returns
    -------
    numpy.float64
        Single aggregated numerical value for the bucket.

    Raises
    ------
    NotImplementedError
        Asserts if supplied input strategy option is not recognised (implemented).

    """
    if strategy == "mean":
        return values.mean()
    if strategy == "median":
        return values.median()

    raise NotImplementedError(f"Provided {strategy} is not a valid one!")


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

    """
    log_counts_step(data, "Gathering event features...", columns=True)
    data = filter_upto_window(data, window=aggregator.window)
    log_counts_step(data, "Filtering events within window...", columns=True)
    event_names = list(data[EVENT_NAME].unique())

    columns = [ENCOUNTER_ID, TIMESTEP] + event_names
    features = pd.DataFrame(columns=columns)

    grouped_events = data.groupby([ENCOUNTER_ID])
    for encounter_id, events in tqdm(grouped_events):
        events[TIMESTEP] = (
            events[EVENT_TIMESTAMP] - min(events[EVENT_TIMESTAMP])
        ) / pd.Timedelta(hours=aggregator.bucket_size)
        events[TIMESTEP] = events[TIMESTEP].astype("int")
        num_timesteps = max(events[TIMESTEP].unique())

        for timestep in range(num_timesteps + 1):
            events_values_timestep = pd.Series(
                [np.nan for _ in range(len(event_names))], index=event_names
            )
            events_timestep = events[events[TIMESTEP] == timestep]
            grouped_events_timestep = events_timestep.groupby([EVENT_NAME])

            for event_name, event_timestep in grouped_events_timestep:
                events_values_timestep[event_name] = aggregate_values_in_bucket(
                    event_timestep[EVENT_VALUE], strategy=aggregator.strategy
                )
            events_values_timestep = pd.DataFrame(
                [[encounter_id, timestep, *events_values_timestep]],
                columns=columns,
            )
            features = pd.concat([features, events_values_timestep])

    features = features.reset_index()
    features = features.set_index([ENCOUNTER_ID, TIMESTEP])

    return features


@time_function
def gather_static_features(data: pd.DataFrame) -> pd.DataFrame:
    """Gathers encounter specific static features.

    Patient statics gathered into features. This function groups patient static
    information, checks to see if there is a unique static value for that given
    encounter, and creates a feature column with that unique value.

    Parameters
    ----------
    data: pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame:
        Processed static features.

    """
    log_counts_step(data, "Gathering static features...", columns=True)
    encounters = list(data[ENCOUNTER_ID].unique())
    col_names = [col_name for col_name in data.columns if col_name != ENCOUNTER_ID]
    features = pd.DataFrame(index=encounters, columns=col_names)

    grouped = data.groupby([ENCOUNTER_ID])
    for encounter_id, statics in grouped:
        for col_name in col_names:
            if statics[col_name].nunique() != 1:
                LOGGER.warning(
                    """None or Duplicate values encountered in patient statics,
                    in %s column, skipped for processing!""",
                    col_name,
                )
                continue
            features.loc[encounter_id, col_name] = statics[col_name].unique()[0]

    return features
