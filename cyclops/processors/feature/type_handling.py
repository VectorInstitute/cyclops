"""Detect and convert feature types."""

"""
CHECK VALID, CONVERT DATATYPE


ordinal, binary - save mapping from category to actual
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from cyclops.processors.constants import (
    NUMERIC,
    BINARY,
    STRING,
    ORDINAL,
    CATEGORICAL_INDICATOR,
    FEATURE_TYPES
)

# PLACEHOLDER
def fn():
    return None


def get_unique(
    values: Union[np.ndarray, pd.Series],
    unique: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get the unique values of an array or series.
    
    The utility of this function comes from checking whether the
    unique values have already been calculated. This function
    assumes that if unique is passed in, it is correct.
    
    Parameters
    ----------
    values: numpy.ndarray or pandas.Series
        Values for which to get the unique values.
    unique: numpy.ndarray, default = None
        If None, calculate the unique values, otherwise take
        and return this parameter as the unique values.
    
    Returns
    -------
    numpy.ndarray
        The unique values.
    """
    if unique is None:
        return values.unique()
    else:
        return unique


def valid_feature_type(type_: str, raise_error: bool = True):
    """Check whether a feature type name is a valid type.
    
    Parameters
    ----------
    type_: str
        Feature type name.
    raise_error: bool, default = True
        Whether to raise an error is the type is not valid.
    
    Returns
    -------
    bool
        Whether the type is valid.
    """
    if type_ in FEATURE_TYPES:
        return True
    
    if raise_error:
        raise ValueError(
            f"Feature type '{type_}' not in {', '.join(FEATURE_TYPES)}."
        )
    
    return False


def type_to_dtype(
    type_: str,
    series: Optional[pd.Series] = None,
    unique: Optional[np.ndarray] = None,
) -> Optional[type]:
    """Determine the datatype for each feature type.
    
    Can optionally specify the feature data for more specific
    datatype selection.
    
    Parameters
    ----------
    type_: str
        Feature type.
    series: pandas.Series, default = None
        Feature data.
    unique: numpy.ndarray, default = None
        Unique feature values.
    
    Returns
    -------
    type
        The feature's datatype.

    """
    # If numeric, leave as is - the user should choose the precision.
    if type_ == STRING:
        return str
    if type_ == NUMERIC:
        return None
    elif type_ == BINARY or type_ == CATEGORICAL_INDICATOR:
        return np.uint8
    elif type_ == ORDINAL:
        if series is None:
            return np.unint8
        else:
            unique = get_unique(series, unique=unique)
            if len(unique) < 255:
                return np.uint8
            else:
                return np.uint16
    
    if valid_feature_type(type_, raise_error=True):
        raise ValueError("Supported type has no corresponding datatype.")


def to_dtype(
    series: Optional[pd.Series],
    type_: str,
    unique: Optional[np.ndarray] = None,
) -> Optional[type]:
    """Set the proper datatype according to feature type.
    
    Parameters
    ----------
    type_: str
        Feature type.
    series: pandas.Series, default = None
        Feature data.
    unique: numpy.ndarray, default = None
        Unique feature values.
    
    Returns
    -------
    pandas.Series
        The feature with a proper datatype.

    """
    dtype = type_to_dtype(type_, series=series, unique=unique)
    if dtype is None:
        return series
    elif series.dtype == dtype:
        return series
    else:
        return series.astype(dtype)


def to_string(
    series: pd.Series,
    unique: Optional[np.ndarray] = None
) -> pd.Series:
    return to_dtype(series, STRING, unique=unique), {}


def to_ordinal(
    series: pd.Series,
    as_binary = False,
    unique: Optional[np.ndarray] = None,
) -> tuple:
    """
    
    Returns
    -------
    tuple
        Return types (pd.Series, dict) with meaning (data, metadata).
    """
    unique = get_unique(series, unique)
    unique = unique.sort()

    map_dict: dict = {}
    for i, unique_val in enumerate(unique):
        map_dict[unique_val] = i

    series = series.replace(map_dict, inplace=inplace)
    
    meta = {"mapping": map_dict}
    
    if as_binary:
        return to_dtype(series, BINARY, unique=unique), meta
    else:
        return to_dtype(series, ORDINAL, unique=unique), meta


def to_binary(
    series: pd.Series,
    unique: Optional[np.ndarray] = None
) -> pd.Series:
    unique = get_unique(values, unique)
    
    if len(unique) != 2:
        raise ValueError(
            "There are more than two unique values. Binary features must have two unique values, e.g., [0, 1], ['A', 'B']."
        )

    return to_ordinal(series, as_binary=True, unique=unique)


def to_numeric(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    return pd.to_numeric(series)


def to_categorical_indicators(
    data: pd.DataFrame,
    col: str,
    unique: Optional[np.ndarray] = None,
    raise_error_many_categories: bool = True,
) -> pd.DataFrame:
    """Convert to binary categorical indicators, i.e., one-hot encode."""
    series = data[col]
    unique = get_unique(series, unique=unique)
    
    if len(unique) > 100 and raise_error_many_categories:
        raise ValueError("Too many categories to convert to categorical.")
    
    series = series.fillna(MISSING_CATEGORY)
    
    dummies = pd.get_dummies(series, prefix=series.name)
    
    intersect = set(dummies.columns).intersection(data.columns)
    if len(intersect) > 0:
        raise ValueError(f"Cannot duplicate columns {', '.join(intersect)}.")
    
    data = pd.concat([data, dummies], axis=1)
    data = data.drop([feature_to_encode], axis=1)
    
    added_meta = {}
    for col in feature_to_encode.columns:
        added_meta[col] = {
            type_: CATEGORICAL_INDICATOR,
            target: False,
            indicator_of: col,
        }
    
    return data, added_meta


def attempt_to_numeric(
    series: pd.Series,
    raise_error: bool = True
) -> tuple:
    if raise_error:
        series = pd.to_numeric(series)
        converted = True
    else:
        try:
            series = pd.to_numeric(series)
            converted = True
        except (ValueError, TypeError):
            converted = False

    return series, converted


CONVERSIONS = {
    NUMERIC: {
        (BINARY, to_binary),
        (STRING, to_string),
        (ORDINAL, to_ordinal),
        (CATEGORICAL_INDICATOR, fn),
    },
    BINARY: {
        (NUMERIC, fn),
        (STRING, to_string),
    },
    STRING: {
        (NUMERIC, fn),
        (ORDINAL, fn),
        (CATEGORICAL_INDICATOR, fn),
        (STRING, to_string),
    },
    ORDINAL: {
        (NUMERIC, fn),
        (CATEGORICAL_INDICATOR, fn),
        (STRING, to_string),
    },
    CATEGORICAL_INDICATOR: {},
}


def valid_binary(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    raise_error: bool = False,
) -> bool:
    unique = get_unique(series, unique=unique)
    valid = set(unique) == set([0, 1])
    
    if not valid and raise_error:
        raise ValueError("Not a valid binary feature.")
    
    return valid


def valid_ordinal(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    raise_error: bool = False,
) -> bool:
    unique = get_unique(series, unique=unique)
    unique = np.sort(unique)
    valid = np.array_equal(unique, np.arange(0, len(unique)))

    if not valid and raise_error:
        raise ValueError("Not a valid ordinal feature.")

    return valid


def values_allowed_by_type(series: pd.Series, type_):
    """Check that features values are valid given its type.
    
    Parameters
    ----------
    series: pandas.Series
        Feature.
    type_: str
        Feature type.
    """
    if type_ == BINARY or type_ == CATEGORICAL_INDICATOR:
        return allowed_binary(series)
    elif type_ == ORDINAL:
        return allowed_ordinal(series)
    
    # Any remaining types do not have value restrictions
    # Just do a check to make sure the type itself is valid
    return valid_feature_type(type_, raise_error=True)


def normalize_types(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the types...
    
    """
    data = data.infer_objects()
    
    # Attempt columns to numeric.
    for col in data:
        data[col], converted = attempt_to_numeric(data[col], raise_error=False)

    return data


def detect_type(series: pd.Series):
    if is_string_dtype(series):
        return STRING

    # Must be either string or numeric
    if not is_numeric_dtype(series):
        raise ValueError(f"Cannot detect data type of column '{series.name}'.")

    # Check whether all values are NaN
    if series.isnull().all():
        raise ValueError(
            f"Cannot detect data type of all null column '{series.name}'."
        )

    unique = get_unique(series)

    if valid_binary(series, unique=unique):
        return BINARY
    
    if valid_ordinal(series, unique=unique):
        return ORDINAL

    return NUMERIC


def detect_types(data: pd.DataFrame):
    types = {}
    for col in data.columns:
        types[col] = detect_type(data[col])
    
    return types

def to_types(
    data: pd.DataFrame,
    old_types: dict,
    new_types: dict
) -> pd.DataFrame:
    """
    for col, old_type in old_types.items():
        new_type = new_types[col]
        
        valid_feature_type(type_, raise_error=True)
        
        valid_conversions = CONVERSIONS[old_type]
        new_types[col]
        data[key] = TRANSFORM_FUNCTION(data[key])
    """
    return None


def infer_types(
    data: pd.DataFrame,
    prev_types: Optional[dict] = None,
    allow_categorical: bool = True
):
    """Infer intended feature types and perform the relevant conversions.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Feature data.
    prev_types: dict
        A map from the column name to the feature type.
    allow_categorical:
        Allow conversion to categorical indicators.
        Otherwise, features will be inferred as ordinal.
    
    Returns
    -------
    pandas.DatFrame
        Updated feature data.
    
    """
    
    """
    def infer_type(series: pd.Series, prev_type):
        
        #convertible_to_binary
        #valid_binary
        
        
        try:
            return to_binary(), BINARY
        except:
            pass
        
        if allow_categorical:
            pass

        # If all values are NaN, feature is dropped.
        if len(unique) == 1 and np.isnan(unique[0]):
            LOGGER.warning("Feature %s has all NaNs, will not be added.", col)
            

        # Check if it can be represented as binary.
        # (Numeric or string alike)
        if len(unique) == 2:
            # If one of them NaN, don't convert to binary, instead numerical.
            min_unique = np.min(unique)
            if not isinstance(min_unique, str) and np.isnan(min_unique):
                self._add_numerical(data[col], aggregate_type=aggregate_type)
                continue
            # Add as binary.
            self._add_binary(data[col], aggregate_type=aggregate_type)
            continue

        # Check for and add numerical features.
        if is_numeric_dtype(data[col]):
            self._add_numerical(data[col], aggregate_type=aggregate_type)
            continue

        # Check for (non-binary valued) string types.
        if is_string_dtype(data[col]) and len(unique) < 100:
            self._add_categorical(data[col], aggregate_type=aggregate_type)
            continue

        LOGGER.warning("Unsure about column %s data type, will not be added", col)

    for col in data.columns:
        infer_type(data[col], prev_types[col])
    """
    return None