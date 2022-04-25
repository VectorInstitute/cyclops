"""Vitals processor module."""

# mypy: ignore-errors

import logging

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import EVENT_NAME, EVENT_VALUE, EVENT_VALUE_UNIT
from cyclops.processors.constants import NEGATIVE_RESULT_TERMS, POSITIVE_RESULT_TERMS
from cyclops.processors.string_ops import (
    fill_missing_with_nan,
    fix_inequalities,
    remove_text_in_parentheses,
    replace_if_string_match,
    strip_whitespace,
    to_lower,
)
from cyclops.processors.utils import log_counts_step
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
        data, "Remove text in parentheses and normalize lab test names...", columns=True
    )

    return data


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
    data = data.copy()
    data[EVENT_VALUE] = data[EVENT_VALUE].apply(fix_inequalities)
    log_counts_step(
        data, "Fixing inequalities and removing outlier values...", columns=True
    )

    data[EVENT_VALUE] = data[EVENT_VALUE].apply(
        replace_if_string_match, args=("|".join(POSITIVE_RESULT_TERMS), "1")
    )
    data[EVENT_VALUE] = data[EVENT_VALUE].apply(
        replace_if_string_match, args=("|".join(NEGATIVE_RESULT_TERMS), "0")
    )
    log_counts_step(data, "Convert Positive/Negative to 1/0...", columns=True)

    data[EVENT_VALUE] = data[EVENT_VALUE].apply(fill_missing_with_nan)
    log_counts_step(data, "Fill empty result string values with NaN...", columns=True)

    data[EVENT_VALUE] = data[EVENT_VALUE].astype("float")
    LOGGER.info("Converting string result values to numeric...")

    return data


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
    LOGGER.info("Cleaning units and converting to SI...")
    data[EVENT_VALUE_UNIT] = data[EVENT_VALUE_UNIT].apply(to_lower)
    data[EVENT_VALUE_UNIT] = data[EVENT_VALUE_UNIT].apply(strip_whitespace)

    return data


@time_function
def clean_events(data) -> pd.DataFrame:
    """Pre-process and clean raw event data.

    Parameters
    ----------
    data: pandas.DataFrame
        Raw event data.

    Returns
    -------
    pandas.DataFrame
        Cleaned event data.

    """
    log_counts_step(data, "Cleaning raw event data...", columns=True)
    data = drop_unsupported(data)
    data = normalize_names(data)
    data = normalize_values(data)

    if EVENT_VALUE_UNIT in list(data.columns):
        data = normalize_units(data)

    return data
