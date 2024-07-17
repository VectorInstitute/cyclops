"""Events processor module, applies cleaning step to event data before aggregation."""

import logging
from typing import List, Optional, Union

import pandas as pd

from cyclops.data.constants import (
    NEGATIVE_RESULT_TERMS,
    POSITIVE_RESULT_TERMS,
)
from cyclops.data.string_ops import (
    fill_missing_with_nan,
    fix_inequalities,
    none_to_empty_string,
    remove_text_in_parentheses,
    replace_if_string_match,
    strip_whitespace,
    to_lower,
)
from cyclops.data.utils import log_df_counts
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def normalize_names(names: pd.Series) -> pd.Series:
    """Normalize column names such that they can be used as features.

    Perform basic cleaning/house-keeping of column names.
    e.g. remove parentheses from the measurement-name,
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
    return names.str.strip()


def normalize_values(values: pd.Series) -> pd.Series:
    """Normalize value columns such that they can be used as features.

    Perform basic cleaning/house-keeping of column values.
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
        replace_if_string_match,
        args=("|".join(POSITIVE_RESULT_TERMS), "1"),
    )
    values = values.apply(
        replace_if_string_match,
        args=("|".join(NEGATIVE_RESULT_TERMS), "0"),
    )

    values = values.apply(remove_text_in_parentheses)
    values = values.apply(fix_inequalities)
    values = values.apply(fill_missing_with_nan)
    return values.astype("float")


def normalize_units(units: pd.Series) -> pd.Series:
    """Normalize event value units.

    Perform basic cleaning/house-keeping of event value units.

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
    return units.apply(strip_whitespace)


@time_function
def normalize_events(
    data: pd.DataFrame,
    event_name_col: str,
    event_value_col: Optional[str] = None,
    event_value_unit_col: Optional[str] = None,
) -> pd.DataFrame:
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

    log_df_counts(data, event_name_col, "Cleaning raw event data...", columns=True)
    data = data.infer_objects()
    data[event_name_col] = normalize_names(data[event_name_col])

    if event_value_col and data[event_value_col].dtypes == object:  # noqa: E721
        data[event_value_col] = normalize_values(data[event_value_col])
        log_df_counts(data, event_name_col, "Normalized values...", columns=True)

    if event_value_unit_col and event_value_unit_col in list(data.columns):
        data[event_value_unit_col] = normalize_units(data[event_value_unit_col])

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
            "Dropped nulls over columns: %s. Removed %d rows.",
            "".join(cols),
            num,
        )

    return data
