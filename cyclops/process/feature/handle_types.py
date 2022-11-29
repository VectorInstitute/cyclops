"""Handling feature types."""

# pylint: disable=too-many-lines

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from cyclops.process.constants import (
    BINARY,
    CATEGORICAL_INDICATOR,
    FEATURE_INDICATOR_ATTR,
    FEATURE_MAPPING_ATTR,
    FEATURE_TYPE_ATTR,
    FEATURE_TYPES,
    MISSING_CATEGORY,
    NUMERIC,
    ORDINAL,
    STRING,
)
from cyclops.utils.common import to_list


def get_unique(
    values: Union[np.ndarray, pd.Series],
    unique: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get the unique values of pandas series.

    The utility of this function comes from checking whether the
    unique values have already been calculated. This function
    assumes that if the unique values are passed, they are correct.

    Parameters
    ----------
    values: pandas.Series
        Values for which to get the unique values.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    numpy.ndarray
        The unique values.

    """
    if unique is None:
        return values.unique()

    return unique


def valid_feature_type(type_: str, raise_error: bool = True) -> bool:
    """Check whether a feature type name is valid.

    Parameters
    ----------
    type_: str
        Feature type name.
    raise_error: bool, default = True
        Whether to raise an error is the type is invalid.

    Returns
    -------
    bool
        Whether the type is valid.

    """
    if type_ in FEATURE_TYPES:
        return True

    if raise_error:
        raise ValueError(f"Feature type '{type_}' not in {', '.join(FEATURE_TYPES)}.")

    return False


def _type_to_dtype(type_: str) -> Optional[Union[type, str]]:
    """Get the Pandas datatype for a feature type name.

    Parameters
    ----------
    type_: str
        Feature type name.

    Returns
    -------
    type or str or None
        The feature's Pandas datatype, or None if no data type
        conversion is desired.

    """
    if type_ == STRING:
        # If string, leave as is - the user can choose the specific length/type.
        # return str
        return None

    if type_ == NUMERIC:
        # If numeric, leave as is - the user can choose the precision.
        return None

    if type_ in (BINARY, CATEGORICAL_INDICATOR):
        # return np.uint8
        return "category"

    if type_ == ORDINAL:
        return "category"

    # Check first if the type is valid, if so, then it isn't supported in this function.
    if valid_feature_type(type_, raise_error=True):
        raise ValueError("Supported type has no corresponding datatype.")

    return None


def to_dtype(series: pd.Series, type_: str) -> pd.Series:
    """Set the series datatype according to the feature type.

    Parameters
    ----------
    type_: str
        Feature type name.
    series: pandas.Series, default = None
        Feature data.

    Returns
    -------
    pandas.Series
        The feature with the corresponding datatype.

    """
    dtype = _type_to_dtype(type_)

    if dtype is None:
        return series

    if series.dtype == dtype:
        return series

    return series.astype(dtype)


def _valid_string(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    raise_error: bool = False,
) -> bool:
    """Check whether a feature is a valid string type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if invalid.

    Returns
    -------
    bool
        Whether the feature is a valid type.

    """
    if is_string_dtype(series):
        return True

    if raise_error:
        raise ValueError("Not a valid string feature.")
    return False


def _convertible_to_string(
    series: pd.Series,  # pylint: disable=unused-argument
    unique: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    raise_error: bool = False,  # pylint: disable=unused-argument
) -> bool:
    """Check whether a feature can be converted to type string.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if the type cannot be converted.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    return True


def _to_string(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Convert type to string.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.

    """
    convertible_to_type(series, STRING, unique=unique, raise_error=True)
    return to_dtype(series, STRING), {FEATURE_TYPE_ATTR: STRING}


def _valid_numeric(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    raise_error: bool = False,
) -> bool:
    """Check whether a feature is a valid numeric type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if invalid.

    Returns
    -------
    bool
        Whether the feature is a valid type.

    """
    if is_numeric_dtype(series):
        return True

    if raise_error:
        raise ValueError("Not a valid string feature.")

    return False


def _convertible_to_numeric(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,  # pylint: disable=unused-argument
    raise_error: bool = False,
) -> bool:
    """Check whether a feature can be converted to type numeric.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if the type cannot be converted.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    if raise_error:
        pd.to_numeric(series)
        return True

    try:
        pd.to_numeric(series)
        can_convert = True
    except (ValueError, TypeError):
        can_convert = False

    return can_convert


def _to_numeric(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Convert type to numeric.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.

    """
    convertible_to_type(series, NUMERIC, unique=unique, raise_error=True)
    series = pd.to_numeric(series)
    return to_dtype(series, NUMERIC), {FEATURE_TYPE_ATTR: NUMERIC}


def _convertible_to_categorical(  # pylint: disable=too-many-arguments
    series: pd.Series,
    category_min: int = None,
    category_max: int = None,
    unique: Optional[np.ndarray] = None,
    raise_error_over_max: bool = False,
    raise_error_under_min: bool = False,
) -> bool:
    """Check whether a feature can be converted to some categorical type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    category_min: int, optional
        The minimum number of categories allowed.
    category_max: int, optional
        The maximum number of categories allowed.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error_over_max: bool, default = False
        Whether to raise an error if there are more categories than max.
    raise_error_under_min: bool, default = False
        Whether to raise an error if there are less categories than min.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    # If numeric, only allow conversion if an integer type
    if is_numeric_dtype(series) and not is_integer_dtype(series):
        return False

    unique = get_unique(series, unique=unique)
    nonnull_unique = unique[~pd.isnull(unique)]
    nunique = len(nonnull_unique)

    if category_min is None:
        min_cond = True
    else:
        min_cond = nunique >= category_min

    if category_max is None:
        max_cond = True
    else:
        max_cond = nunique <= category_max

    # Convertible
    if min_cond and max_cond:
        return True

    # Not convertible
    if max_cond and raise_error_over_max:
        raise ValueError(
            f"Should have at most {category_max} categories, but has {nunique}."
        )

    if min_cond and raise_error_under_min:
        raise ValueError(
            f"Should have at least {category_min} categories, but has {nunique}."
        )

    return False


def _valid_ordinal(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    raise_error: bool = False,
) -> bool:
    """Check whether a feature is a valid ordinal type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if invalid.

    Returns
    -------
    bool
        Whether the feature is a valid type.

    """
    unique = get_unique(series, unique=unique)
    unique = np.sort(unique)
    valid = np.array_equal(unique, np.arange(0, len(unique)))

    if valid:
        return True

    if raise_error:
        raise ValueError("Not a valid ordinal feature.")

    return False


def _convertible_to_ordinal(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    category_max: int = 20,
    raise_error_over_max: bool = False,
) -> bool:
    """Check whether a feature can be converted to type ordinal.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    category_max: int, optional
        The number of categories above which the feature is not considered ordinal.
    raise_error_over_max: bool, default = False
        Whether to raise an error if there are more categories than max.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    return _convertible_to_categorical(
        series,
        category_min=2,
        category_max=category_max,
        unique=unique,
        raise_error_over_max=raise_error_over_max,
    )


def _numeric_categorical_mapping(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Map values to categories in a series.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.

    """
    unique = get_unique(series, unique=unique)
    if unique.dtype.name == "object":
        unique = unique.astype(str)

    unique.sort()

    map_dict: dict = {}
    for i, unique_val in enumerate(unique):
        map_dict[unique_val] = i

    series = series.replace(map_dict)

    inv_map = {v: k for k, v in map_dict.items()}
    meta = {FEATURE_MAPPING_ATTR: inv_map}

    return series, meta


def _to_ordinal(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Convert type to ordinal.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.

    """
    series, meta = _numeric_categorical_mapping(series, unique=unique)
    meta[FEATURE_TYPE_ATTR] = ORDINAL
    return to_dtype(series, ORDINAL), meta


def _valid_binary(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    raise_error: bool = False,
) -> bool:
    """Check whether a feature is a valid binary type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if invalid.

    Returns
    -------
    bool
        Whether the feature is a valid type.

    """
    unique = get_unique(series, unique=unique)
    nonnull_unique = unique[~pd.isnull(unique)]
    valid = set(nonnull_unique) == set([0, 1])

    if valid:
        return True

    if raise_error:
        raise ValueError("Not a valid binary feature.")

    return False


def _convertible_to_binary(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> bool:
    """Check whether a feature can be converted to type binary.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    if is_bool_dtype(series):
        return True

    return _convertible_to_categorical(
        series,
        category_min=2,
        category_max=2,
        unique=unique,
    )


def _to_binary(
    series: pd.Series, unique: Optional[np.ndarray] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Convert type to binary.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.Series, dict) with the updated feature data
        and metadata respectively.

    """
    if is_bool_dtype(series):
        meta = {
            FEATURE_TYPE_ATTR: BINARY,
            FEATURE_MAPPING_ATTR: {False: False, True: True},
        }
        return to_dtype(series, BINARY), meta

    series, meta = _numeric_categorical_mapping(series, unique=unique)
    meta[FEATURE_TYPE_ATTR] = BINARY
    return to_dtype(series, BINARY), meta


def _valid_categorical_indicator(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    raise_error: bool = False,
) -> bool:
    """Check whether a feature is a valid categorical indicator type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.
    raise_error: bool, default = False
        Whether to raise an error if invalid.

    Returns
    -------
    bool
        Whether the feature is a valid type.

    """
    valid = is_valid(series, CATEGORICAL_INDICATOR, unique=unique)

    if valid:
        return True

    if raise_error:
        raise ValueError("Not a valid string feature.")

    return False


def _convertible_to_categorical_indicators(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
    category_max: int = 20,
    raise_error_over_max: bool = False,
) -> bool:
    """Check whether a feature can be converted to categorical indicators.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    return _convertible_to_categorical(
        series,
        category_min=2,
        category_max=category_max,
        unique=unique,
        raise_error_over_max=raise_error_over_max,
    )


def _to_categorical_indicators(
    data: pd.DataFrame,
    col: str,
    unique: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert type to binary categorical indicators.

    This performs the Pandas equivalent of one-hot encoding.

    Parameters
    ----------
    data: pandas.DataFrame
        Features data.
    col: str
        Feature column being converted.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.DataFrame, dict) with the updated features data
        and metadata respectively.

    """
    series = data[col]
    unique = get_unique(series, unique=unique)

    # series = series.fillna(MISSING_CATEGORY)

    dummies = pd.get_dummies(series, prefix=series.name)

    meta = {}
    for dummy_col in dummies.columns:
        dummies[dummy_col] = to_dtype(dummies[dummy_col], CATEGORICAL_INDICATOR)
        meta[dummy_col] = {
            FEATURE_TYPE_ATTR: CATEGORICAL_INDICATOR,
            FEATURE_INDICATOR_ATTR: col,
        }

    intersect = set(dummies.columns).intersection(data.columns)
    if len(intersect) > 0:
        raise ValueError(f"Cannot duplicate columns {', '.join(intersect)}.")

    data = pd.concat([data, dummies], axis=1)
    data = data.drop([col], axis=1)

    return data, meta


def convertible_to_type(
    series: pd.Series,
    type_: str,
    unique: np.ndarray = None,
    raise_error: bool = False,
) -> bool:
    """Check whether a feature can be converted to some type.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    type_: str
        Feature type name to check for conversion.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    bool
        Whether the feature can be converted.

    """
    if type_ == NUMERIC:
        convertible = _convertible_to_numeric(series, unique=unique)

    elif type_ == STRING:
        convertible = _convertible_to_string(series, unique=unique)

    elif type_ == BINARY:
        convertible = _convertible_to_binary(series, unique=unique)

    elif type_ == ORDINAL:
        convertible = _convertible_to_ordinal(series, unique=unique)

    elif type_ == CATEGORICAL_INDICATOR:
        convertible = _convertible_to_categorical_indicators(series, unique=unique)

    else:
        # Check first if the type is valid, if so, then it isn't supported here.
        if valid_feature_type(type_, raise_error=True):
            raise ValueError("Supported type has no corresponding datatype.")

    if raise_error and not convertible:
        raise ValueError(f"Cannot convert series {series.name} to type {type_}.")

    return convertible


def is_valid(
    series: pd.Series,
    type_: str,
    unique: Optional[np.ndarray] = None,
) -> bool:
    """Check whether a feature is valid as a given type.

    Parameters
    ----------
    series: pandas.Series
        Feature.
    type_: str
        Feature type name.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    bool
        Whether the feature is valid.

    """
    if type_ in (BINARY, CATEGORICAL_INDICATOR):
        return _valid_binary(series, unique=unique)

    if type_ == ORDINAL:
        return _valid_ordinal(series, unique=unique)

    if type_ == STRING:
        return _valid_string(series, unique=unique)

    if type_ == NUMERIC:
        return _valid_numeric(series, unique=unique)

    # Any remaining types do not have validity functions
    # If the type passed is valid, then return True
    return valid_feature_type(type_, raise_error=True)


def normalize_data(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Normalize feature data to more easily deal with types.

    Parameters
    ----------
    data: pandas.DataFrame
        Features data.
    features: list of str
        Features to normalize.

    Returns
    -------
    pandas.DataFrame
        Updated features data.

    """
    data = data.infer_objects()

    # Attempt to convert columns to numeric.
    numeric_map = {
        col: NUMERIC for col in features if convertible_to_type(data[col], NUMERIC)
    }
    data, _ = to_types(data, numeric_map)

    for col in data:
        if is_string_dtype(data[col]):
            data[col] = data[col].astype(str)
            data[col] = data[col].replace("None", np.nan)
            # NAN_SUBSTITUTION_VALUE
            # data[col] = data[col].str.encode('utf-8').astype('|S')
            # .astype('|S')

    return data


def _to_type(
    data: pd.DataFrame,
    col: str,
    new_type: str,
    unique: Optional[np.ndarray] = None,
) -> Tuple[Union[pd.Series, pd.DataFrame], Dict[str, Any]]:
    """Convert a feature to a given type.

    Parameters
    ----------
    data: pandas.DataFrame
        Features data.
    col: str
        Column name for the feature being converted.
    new_type: str
        Feature type name of type to which to convert.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    tuple
        Tuple (pandas.Series or pandas.DataFrame, dict) with the updated
        features data and metadata respectively. If converting to categorical
        indicators, a DataFrame is returned, otherwise a Series is returned.

    """
    if new_type == CATEGORICAL_INDICATOR:
        if data is None:
            raise ValueError(
                "The features data must be passed to keyword argument 'data'."
            )
        return _to_categorical_indicators(data, col, unique=unique)

    if new_type == STRING:
        series, meta = _to_string(data[col], unique=unique)

    elif new_type == ORDINAL:
        series, meta = _to_ordinal(data[col], unique=unique)

    elif new_type == BINARY:
        series, meta = _to_binary(data[col], unique=unique)

    elif new_type == NUMERIC:
        series, meta = _to_numeric(data[col], unique=unique)

    else:
        # Check if an incorrect type was passed, otherwise
        # say that it isn't supported.
        if valid_feature_type(new_type, raise_error=True):
            raise ValueError(f"Cannot convert to type {new_type}.")

    data[col] = series
    meta = {series.name: meta}
    return data, meta


def to_types(
    data: pd.DataFrame,
    new_types: dict,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert features to given types.

    Parameters
    ----------
    data: pandas.DataFrame
        Features data.
    new_types: str
        Map from the feature column name to its new type.

    Returns
    -------
    tuple
        Tuple (pandas.DataFrame, dict) with the updated features data
        and metadata respectively.

    """
    meta = {}
    for col, new_type in new_types.items():
        data, fmeta = _to_type(data, col, new_type)
        meta.update(fmeta)

    return data, meta


def _infer_type(
    series: pd.Series,
    unique: Optional[np.ndarray] = None,
) -> str:
    """Infer intended feature type and perform the relevant conversion.

    Parameters
    ----------
    series: pandas.Series
        Feature data.
    unique: numpy.ndarray, optional
        Unique values which can be optionally specified.

    Returns
    -------
    str
        Feature type name.

    """
    unique = get_unique(series, unique=unique)

    if convertible_to_type(series, BINARY, unique=unique):
        return BINARY

    if convertible_to_type(series, ORDINAL, unique=unique):
        return ORDINAL

    if convertible_to_type(series, NUMERIC, unique=unique):
        return NUMERIC

    if convertible_to_type(series, STRING, unique=unique):
        return STRING

    raise ValueError(f"Could not infer type of series '{series.name}'.")


def infer_types(
    data: pd.DataFrame,
    features: List[str],
) -> Dict[str, str]:
    """Infer intended feature types and perform the relevant conversions.

    Parameters
    ----------
    data: pandas.DataFrame
        Feature data.
    features: list of str
        Features to consider.

    Returns
    -------
    tuple
        A tuple (pandas.DataFrame, dict) with the updated features data
        and metadata respectively.

    """
    new_types = {}
    for col in features:
        new_types[col] = _infer_type(data[col])

    return new_types


def collect_indicators(
    data: pd.DataFrame,
    categorical: Union[str, List[str]],
) -> pd.DataFrame:
    """Infer and collect indicator features into ordinal categorical features.

    Parameters
    ----------
    data: pandas.DataFrame
        Features data.
    categorical: str or list of str
        Names of categorical features. E.g., "hospital" for indicators
        "hospital_A", "hospital_B"

    """
    categorical = to_list(categorical)

    meta = {}
    for cat in categorical:
        indicators = [
            name
            for name in data.columns
            if name.startswith(cat + "_") and _valid_binary(data[name])
        ]

        if len(indicators) < 2:
            raise ValueError("Not enough indicators to convert to ordinal.")

        if not (data[indicators].sum(axis=1).values == 1).all():
            raise ValueError("Indicators be converted into ordinal.")

        # Get categories
        data[cat] = np.argmax(data[indicators].values, axis=1)
        indicator_names = [
            indicator[len(cat) + 1 :] for indicator in indicators  # noqa: E203
        ]
        map_dict = {
            i: (name if name != MISSING_CATEGORY else np.nan)
            for i, name in enumerate(indicator_names)
        }

        data[cat] = data[cat].replace(map_dict)
        data = data.drop(indicators, axis=1)

        # Convert to ordinal
        unique = np.array(list(map_dict.values()))
        data[cat], fmeta = _to_ordinal(data[cat], unique=unique)
        meta[cat] = fmeta

    return data, meta
