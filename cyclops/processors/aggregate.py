"""Aggregation functions."""

import logging
from dataclasses import dataclass
from typing import Union

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
from cyclops.processors.utils import log_counts_step
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class Aggregator:
    """Aggregation options for temporal data.

    Attributes
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


def _filter_upto_window_since_admission(
    data: pd.DataFrame, window: int = 24
) -> pd.DataFrame:
    """Filter data based on single time window value.

    For e.g. if window is 24 hrs, then all data for the encounter
    upto after 24 hrs of admission are considered. Useful for
    one-shot prediction.

    Parameters
    ----------
    data: pandas.DataFrame
        Data before filtering.
    window: int, optional
        Window (no. of hrs) upto after admission to consider.

    Returns
    -------
    pandas.DataFrame
        Filtered data frame, with aggregates collected within window.

    """
    data_filtered = data.copy()
    sample_time = data_filtered[EVENT_TIMESTAMP]
    admit_time = data_filtered[ADMIT_TIMESTAMP]
    window_condition = (sample_time - admit_time) / pd.Timedelta(hours=1)
    data_filtered = data_filtered.loc[window_condition <= window]
    return data_filtered


@time_function
def gather_event_features(
    data, aggregator: Aggregator
) -> Union[pd.DataFrame, pd.MultiIndex]:
    """Gather events from encounters into time-series features.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.
    aggregator: cyclops.processor.Aggregator
        Aggregation options.

    Returns
    -------
    pandas.DataFrame or pd.MultiIndex:
        Processed event features.

    """
    log_counts_step(data, "Gathering event features...", columns=True)
    data = _filter_upto_window_since_admission(data, window=aggregator.window)
    log_counts_step(data, "Filtering events within window...", columns=True)
    event_names = list(data[EVENT_NAME].unique())
    encounters = list(data[ENCOUNTER_ID].unique())

    timestep_col = np.zeros(len(data)).astype(int)

    grouped_events = data.groupby([ENCOUNTER_ID, EVENT_NAME])
    index = 0
    for (encounter_id, event_name), events in grouped_events:
        earliest_ts = min(events[EVENT_TIMESTAMP])
        for _, event in events.iterrows():
            event_ts = event[EVENT_TIMESTAMP]
            timestep = (event_ts - earliest_ts) / pd.Timedelta(
                hours=aggregator.bucket_size
            )
            timestep_col[index] = int(timestep)
            index += 1
    data[TIMESTEP] = timestep_col.tolist()

    features = data.copy()
    features = features.reset_index()
    features = features.set_index([ENCOUNTER_ID, TIMESTEP])

    return features


@time_function
def gather_static_features(data) -> pd.DataFrame:
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

    Raises
    ------
    AssertionError
        Asserts to ensure that each static features have a single unique value
        for a patient for the encounter.

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
