"""Events processor module, applies cleaning step to event data before aggregation."""

import logging
from typing import List, Optional, Union

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
)
from cyclops.processors.constants import (
    EMPTY_STRING,
    NEGATIVE_RESULT_TERMS,
    POSITIVE_RESULT_TERMS,
)
from cyclops.processors.string_ops import (
    fill_missing_with_nan,
    fix_inequalities,
    none_to_empty_string,
    remove_text_in_parentheses,
    replace_if_string_match,
    strip_whitespace,
    to_lower,
)
from cyclops.processors.util import assert_has_columns, gather_columns, log_counts_step
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


UNSUPPORTED = [
    "urinalysis",
    "alp",
    "alt",
    "ast",
    "d-dimer",
    "ldh",
    "serum osmolality",
    "tsh",
    "urine osmolality",
]


def combine_events(event_data: Union[pd.DataFrame, List[pd.DataFrame]]) -> pd.DataFrame:
    """Gather event data from multiple dataframes into a single one.

    Events can be in multiple raw dataframes like labs, vitals, etc. This
    function takes in multiple dataframes and gathers all events into a single
    dataframe. If just a single dataframe is passed, it returns it back.

    Parameters
    ----------
    event_data: pandas.DataFrame or list of pandas.DataFrame
        Raw event data.

    Returns
    -------
    pandas.DataFrame
        Combined event data.

    """

    def add_events(events: pd.DataFrame, events_to_add: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                events,
                events_to_add,
            ],
            ignore_index=True,
            axis=0,
        )

    events = pd.DataFrame()
    if isinstance(event_data, pd.DataFrame):
        event_data = [event_data]
    for event_dataframe in event_data:
        events = add_events(events, event_dataframe)

    return events


@assert_has_columns([ENCOUNTER_ID])
def convert_to_events(
    data: pd.DataFrame,
    event_name: str,
    timestamp_col: str,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert dataframe with just timestamps into events.

    Any event like admission, discharge, transfer, etc. can be converted to the
    common events dataframe format with 'encounter_id', 'event_name', 'event_timestamp',
    and 'event_value' columns. The input data in this case does not have an explicit
    event name and hence we assign it. Like for e.g. admission.

    Parameters
    ----------
    data: pandas.DataFrame
        Raw data with some timestamps denoting an event.
    event_name: str
        Event name to give, added as a new column.
    timestamp_col: str
        Name of the column in the incoming dataframe that has the timestamp.
    value_col: str, optional
        Name of the column in the incoming dataframe that has potential event values.

    Returns
    -------
    pandas.DataFrame
        Events in the converted format.

    """
    if value_col:
        cols = [ENCOUNTER_ID, timestamp_col, value_col]
    else:
        cols = [ENCOUNTER_ID, timestamp_col]
    events = gather_columns(data, cols)
    if EVENT_VALUE not in events:
        events[EVENT_VALUE] = EMPTY_STRING
    events = events.rename(
        columns={timestamp_col: EVENT_TIMESTAMP, value_col: EVENT_VALUE}
    )
    events[EVENT_NAME] = event_name

    return events


def is_supported(event_name: str) -> bool:
    """Check if event name is supported.

    Processing events involves data cleaning, and hence some event names
    are currently removed until they are supported.

    Parameters
    ----------
    event_name: str
        Name of event.

    Returns
    -------
    bool
        If supported return True, else False.

    """
    return event_name not in UNSUPPORTED


@assert_has_columns([EVENT_NAME])
def drop_unsupported(data: pd.DataFrame) -> pd.DataFrame:
    """Drop events currently not supported for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame
        Output data with dropped events (rows) which had unsupported events.

    """
    data = data.loc[data[EVENT_NAME].apply(is_supported)]
    log_counts_step(data, "Drop unsupported events...", columns=True)

    return data


@assert_has_columns([EVENT_NAME])
def normalize_names(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize event names.

    Perform basic cleaning/house-keeping of event names.
    e.g. remove parantheses from the measurement-name,
    convert to lower-case.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame
        Output data with normalized event names.

    """
    data[EVENT_NAME] = data[EVENT_NAME].apply(remove_text_in_parentheses)
    data[EVENT_NAME] = data[EVENT_NAME].apply(to_lower)
    log_counts_step(
        data, "Remove text in parentheses and normalize event names...", columns=True
    )

    return data


@assert_has_columns([EVENT_VALUE])
def normalize_values(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize event values.

    Perform basic cleaning/house-keeping of event values.
    e.g. fill empty strings with NaNs, convert some strings to
    equivalent boolean or numeric, so they can be used as features.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame
        Output data with normalized event values.

    """
    data[EVENT_VALUE] = data[EVENT_VALUE].apply(
        replace_if_string_match, args=("|".join(POSITIVE_RESULT_TERMS), "1")
    )
    data[EVENT_VALUE] = data[EVENT_VALUE].apply(
        replace_if_string_match, args=("|".join(NEGATIVE_RESULT_TERMS), "0")
    )
    log_counts_step(data, "Convert Positive/Negative to 1/0...", columns=True)

    data[EVENT_VALUE] = data[EVENT_VALUE].apply(remove_text_in_parentheses)
    log_counts_step(data, "Remove any text in paranthesis", columns=True)

    data[EVENT_VALUE] = data[EVENT_VALUE].apply(fix_inequalities)
    log_counts_step(
        data, "Fixing inequalities and removing outlier values...", columns=True
    )

    data[EVENT_VALUE] = data[EVENT_VALUE].apply(fill_missing_with_nan)
    log_counts_step(data, "Fill empty result string values with NaN...", columns=True)

    data[EVENT_VALUE] = data[EVENT_VALUE].astype("float")
    LOGGER.info("Converting string result values to numeric...")

    return data


@assert_has_columns([EVENT_VALUE_UNIT])
def normalize_units(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize event value units.

    Perform basic cleaning/house-keeping of event value units.
    e.g. converting units to SI.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame
        Output data with normalized event value units.

    """
    LOGGER.info("Normalizing units...")
    data[EVENT_VALUE_UNIT] = data[EVENT_VALUE_UNIT].apply(none_to_empty_string)
    data[EVENT_VALUE_UNIT] = data[EVENT_VALUE_UNIT].apply(to_lower)
    data[EVENT_VALUE_UNIT] = data[EVENT_VALUE_UNIT].apply(strip_whitespace)

    return data


@time_function
def normalize_events(data) -> pd.DataFrame:
    """Pre-process and normalize (clean) raw event data.

    Parameters
    ----------
    data: pandas.DataFrame
        Raw event data.

    Returns
    -------
    pandas.DataFrame
        Cleaned event data.

    """
    data = data.copy()

    log_counts_step(data, "Cleaning raw event data...", columns=True)
    data = data.infer_objects()
    data = normalize_names(data)
    data = drop_unsupported(data)

    if data[EVENT_VALUE].dtypes == object:
        data = normalize_values(data)

    if EVENT_VALUE_UNIT in list(data.columns):
        data = normalize_units(data)

    return data
