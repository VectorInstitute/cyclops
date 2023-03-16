"""Utilities for datasets."""
from typing import List, Union, get_args

from datasets.features import (
    ClassLabel,
    Value,
    Sequence,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
)

NUMERIC_FEATURE_TYPES = (
    "bool",
    "int",
    "uint",
    "float",
    "decimal",
)
DATETIME_FEATURE_TYPES = (
    "date",
    "duration",
    "time",
    "timestamp",
)
FEATURE_TYPES = Union[
    ClassLabel,
    Sequence,
    Value,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
]


def check_required_columns(
    dataset_column_names: List[str], *required_columns: Union[List[str], str, None]
) -> None:
    """Check if required columns are present in dataset.

    Parameters
    ----------
    dataset_column_names : List[str]
        List of column names in dataset.
    required_columns : Union[List[str], str, None]
        List of required column names or single required column name.

    Raises
    ------
    ValueError
        If a required column is not present in the dataset.

    """
    required_columns_ = []
    for required_column in required_columns:
        if required_column is None:
            continue
        if isinstance(required_column, str):
            required_columns_.append(required_column)
        else:
            required_columns_.extend(required_column)

    for column in required_columns_:
        if column is not None and column not in dataset_column_names:
            raise ValueError(
                f"Column {column} is not present in the dataset. Please "
                "specify a valid column. The following columns are present "
                f"in the dataset: {dataset_column_names}."
            )


def feature_is_numeric(feature: FEATURE_TYPES) -> bool:
    """Check if Hugging Face dataset feature is numeric.

    Parameters
    ----------
    feature : Union[ClassLabel, Sequence, Value, Array2D, Array3D, Array4D, Array5D]
        Hugging Face dataset feature.

    Returns
    -------
    bool
        True if feature is numeric, False otherwise.

    Raises
    ------
    TypeError
        If `feature` is not a valid Hugging Face dataset feature.

    """
    dtype = feature.dtype
    if dtype == "list":  # Sequence
        return all(feature_is_numeric(subfeature) for subfeature in feature.feature)
    if not isinstance(feature, get_args(FEATURE_TYPES)):
        raise TypeError(f"Invalid type for `feature`: {type(feature)}.")

    return any(dtype.startswith(t) for t in NUMERIC_FEATURE_TYPES)


def feature_is_datetime(feature: FEATURE_TYPES) -> bool:
    """Check if Hugging Face dataset feature is datetime.

    Parameters
    ----------
    feature : Union[ClassLabel, Sequence, Value, Array2D, Array3D, Array4D, Array5D]
        Hugging Face dataset feature.

    Returns
    -------
    bool
        True if feature is datetime, False otherwise.

    Raises
    ------
    TypeError
        If `feature` is not a valid Hugging Face dataset feature.

    """
    dtype = feature.dtype
    if dtype == "list":  # Sequence
        return all(feature_is_datetime(subfeature) for subfeature in feature.feature)
    if not isinstance(feature, get_args(FEATURE_TYPES)):
        raise TypeError(f"Invalid type for `feature`: {type(feature)}.")

    return any(dtype.startswith(t) for t in DATETIME_FEATURE_TYPES)
