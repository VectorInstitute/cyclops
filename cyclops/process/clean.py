"""Events processor module, applies cleaning step to event data before aggregation."""

import logging
from typing import List, Optional, Union

import pandas as pd

from cyclops.process.column_names import (
    ENCOUNTER_ID,
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
)
from cyclops.process.constants import (
    EMPTY_STRING,
    NEGATIVE_RESULT_TERMS,
    POSITIVE_RESULT_TERMS,
)
from cyclops.process.string_ops import (
    fill_missing_with_nan,
    fix_inequalities,
    none_to_empty_string,
    remove_text_in_parentheses,
    replace_if_string_match,
    strip_whitespace,
    to_lower,
)
from cyclops.process.util import assert_has_columns, gather_columns, log_counts_step
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


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
    event_category: str,
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
    event_category: str
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
    events[EVENT_CATEGORY] = event_category

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


def normalize_names(names: pd.Series) -> pd.Series:
    """Normalize event names.

    Perform basic cleaning/house-keeping of event names.
    e.g. remove parantheses from the measurement-name,
    convert to lower-case.

    Parameters
    ----------
    names: pandas.Series
        Event names.

    Returns
    -------
    pandas.DataFrame
        Normalized event names.

    """
    names = names.apply(to_lower)
    names = names.apply(remove_text_in_parentheses)
    names = names.str.strip()
    return names


def normalize_categories(categories: pd.Series) -> pd.Series:
    """Normalize event category names.

    Perform basic cleaning/house-keeping of event category names.
    e.g. convert to lower-case.

    Parameters
    ----------
    categories: pandas.Series
        Categories.

    Returns
    -------
    pandas.Series
        Normalized event categories.

    """
    categories = categories.apply(to_lower)
    categories = categories.str.strip()
    return categories


def normalize_values(values: pd.Series) -> pd.Series:
    """Normalize event values.

    Perform basic cleaning/house-keeping of event values.
    e.g. fill empty strings with NaNs, convert some strings to
    equivalent boolean or numeric, so they can be used as features.

    Parameters
    ----------
    values: pandas.Series
        Event values.

    Returns
    -------
    pandas.Series
        Normalized event values.

    """
    values = values.apply(
        replace_if_string_match, args=("|".join(POSITIVE_RESULT_TERMS), "1")
    )
    values = values.apply(
        replace_if_string_match, args=("|".join(NEGATIVE_RESULT_TERMS), "0")
    )

    values = values.apply(remove_text_in_parentheses)
    values = values.apply(fix_inequalities)
    values = values.apply(fill_missing_with_nan)
    values = values.astype("float")

    return values


def normalize_units(units: pd.Series) -> pd.Series:
    """Normalize event value units.

    Perform basic cleaning/house-keeping of event value units.
    e.g. converting units to SI.

    Parameters
    ----------
    data: pandas.Series
        Units.

    Returns
    -------
    pandas.Series
        Normalized units.

    """
    LOGGER.info("Normalizing units...")
    units = units.apply(none_to_empty_string)
    units = units.apply(to_lower)
    units = units.apply(strip_whitespace)

    return units


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
    data[EVENT_NAME] = normalize_names(data[EVENT_NAME])
    data = drop_unsupported(data)

    if data[EVENT_VALUE].dtypes == object:
        data[EVENT_VALUE] = normalize_values(data[EVENT_VALUE])
        log_counts_step(data, "Normalized values...", columns=True)

    if EVENT_VALUE_UNIT in list(data.columns):
        data[EVENT_VALUE_UNIT] = normalize_units(data[EVENT_VALUE_UNIT])

    if EVENT_CATEGORY in list(data.columns):
        data[EVENT_CATEGORY] = normalize_categories(data[EVENT_CATEGORY])

    return data


def dropna_rows(data: pd.DataFrame, cols: Union[str, List[str]]) -> pd.DataFrame:
    """Drop null values in some specific columns.

    Part of the utility of this function lies in its logging.

    Parameters
    ----------
    data: pandas.DataFrame
        Data.
    cols: str or list of str
        Columns over which to drop null values.

    Returns
    -------
    pandas.DataFrame
        Data with corresponding null rows dropped.

    """
    length = len(data)
    data = data.dropna(subset=cols)
    new_length = len(data)

    if new_length != length:
        num = length - new_length
        LOGGER.info(
            "Dropped nulls over columns: %s. Removed %d rows.", "".join(cols), num
        )

    return data
