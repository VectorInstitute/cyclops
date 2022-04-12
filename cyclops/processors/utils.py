"""Utility functions for processor API."""

import logging
from typing import Union

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processor import Aggregator
from cyclops.processors.column_names import ADMIT_TIMESTAMP, ENCOUNTER_ID
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


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


@time_function
def gather_event_features(
    data, groupby_col, event_col, aggregator: Aggregator
) -> Union[pd.DataFrame, pd.MultiIndex]:
    """Gather events from encounters into statistical (static) or time-series features.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.
    groupby_col: str
        Name of column to use to group data, e.g. column with lab test names
        can be used to group lab test data.
    event_col: str
        Name of column with event info to use for creating features. e.g. lab
        test result values, or vital measurement value.
    aggregator: cyclops.processor.Aggregator
        Aggregation options.

    Returns
    -------
    pandas.DataFrame or pd.MultiIndex:
        Processed event features.

    """
    log_counts_step(data, "Gathering event features...", columns=True)
    event_names = list(data[groupby_col].unique())
    encounters = list(data[ENCOUNTER_ID].unique())
    features = pd.DataFrame(index=encounters, columns=event_names)

    grouped_events = data.groupby([ENCOUNTER_ID, groupby_col])
    for (encounter_id, event_name), events in grouped_events:
        features.loc[encounter_id, event_name] = events[event_col].mean()

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
    col_names = list(data.columns)
    features = pd.DataFrame(index=encounters, columns=col_names)

    grouped = data.groupby([ENCOUNTER_ID])
    for encounter_id, statics in grouped:
        for col_name in col_names:
            if statics[col_name].nunique() != 1:
                raise AssertionError(
                    """Duplicate values encountered in patient statics,
                    in {col_name} column!"""
                )
            features.loc[encounter_id, col_name] = statics[col_name].unique()[0]

    return features


def check_must_have_columns(data: pd.DataFrame, must_have_columns: list) -> bool:
    """Check if data has required columns for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    must_have_columns: list
        List of column names that must be present in data.

    Returns
    -------
    bool
        True if all required columns are present, else False.

    """
    column_names = list(data.columns)
    return set(must_have_columns).issubset(column_names)


def gather_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Gather specified columns and discard rest.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    columns: list
        List of column names to keep in data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with required columns, other columns discarded.

    """
    return data[columns].copy()


def log_counts_step(data, step_description: str, rows=True, columns=False) -> None:
    """Log num. of encounters and num. of samples (rows).

    Parameters
    ----------
    data: pandas.DataFrame
        Encounter specific input data.
    step_description: str
        Description of intermediate processing step.
    rows: bool
        Log the number of samples, or rows of data.
    columns: bool
        Log the number of data columns.

    """
    LOGGER.info(step_description)
    num_encounters = data[ENCOUNTER_ID].nunique()
    if rows:
        num_samples = len(data)
        LOGGER.info("# samples: %d, # encounters: %d", num_samples, num_encounters)
    if columns:
        num_columns = len(data.columns)
        LOGGER.info("# columns: %d, # encounters: %d", num_columns, num_encounters)
