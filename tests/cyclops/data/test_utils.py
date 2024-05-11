"""Test processor utility functions."""

import numpy as np
import pandas as pd
import pytest
from datasets.features import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Sequence,
    Value,
)

from cyclops.data.utils import (
    create_indicator_variables,
    feature_is_datetime,
    feature_is_numeric,
    gather_columns,
    has_columns,
    has_range_index,
    is_timestamp_series,
    to_range_index,
)


def test_feature_is_numeric():
    """Test feature_is_numeric fn."""
    numeric_features = [
        Value("int32"),
        Value("float64"),
        Array2D(shape=(10, 10), dtype="int32"),
        Array3D(shape=(10, 10, 10), dtype="float64"),
        Array4D(shape=(10, 10, 10, 10), dtype="int32"),
        Array5D(shape=(10, 10, 10, 10, 10), dtype="float64"),
        Sequence([Value("int32"), Value("float64")]),
        Value("bool"),
    ]
    for feature in numeric_features:
        assert feature_is_numeric(
            feature,
        ), f"Failed for {type(feature)} with dtype {feature.dtype}"

    non_numeric_features = [
        Value("string"),
        Array2D(shape=(10, 10), dtype="string"),
    ]
    for feature in non_numeric_features:
        assert not feature_is_numeric(
            feature,
        ), f"Failed for {type(feature)} with dtype {feature.dtype}"

    invalid_features = [None, 123, "invalid", [1, 2, 3], {"key": "value"}]
    for feature in invalid_features:
        with pytest.raises(AttributeError):
            feature_is_numeric(feature)


def test_feature_is_datetime():
    """Test feature_is_datetime fn."""
    datetime_features = [
        Value("timestamp[s]"),
        Array2D(shape=(10, 10), dtype="time[s]"),
        Sequence([Value("timestamp[ns]")]),
    ]
    for feature in datetime_features:
        assert feature_is_datetime(
            feature,
        ), f"Failed for {type(feature)} with dtype {feature.dtype}"

    non_datetime_features = [
        Value("string"),
        Array2D(shape=(10, 10), dtype="string"),
        Value("int32"),
        Array2D(shape=(10, 10), dtype="int32"),
    ]
    for feature in non_datetime_features:
        assert not feature_is_datetime(
            feature,
        ), f"Failed for {type(feature)} with dtype {feature.dtype}"

    invalid_features = [None, 123, "invalid", [1, 2, 3], {"key": "value"}]
    for feature in invalid_features:
        with pytest.raises(AttributeError):
            feature_is_datetime(feature)


def test_create_indicator_variables():
    """Test create_indicator_variables fn."""
    features = pd.DataFrame([[np.nan, 1], [3, np.nan]], columns=["A", "B"])
    indicator_features = create_indicator_variables(features)
    assert (indicator_features.columns == ["A_indicator", "B_indicator"]).all()
    assert (indicator_features["A_indicator"] == pd.Series([0, 1])).all()
    indicator_features = create_indicator_variables(features, columns=["A"])
    assert (indicator_features.columns == ["A_indicator"]).all()
    assert (indicator_features["A_indicator"] == pd.Series([0, 1])).all()


def test_has_columns():
    """Test has_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert has_columns(test_input, ["A", "B", "C"]) is True
    assert has_columns(test_input, ["A", "C"]) is True
    assert has_columns(test_input, ["D", "C"]) is False
    with pytest.raises(ValueError):
        has_columns(test_input, ["D", "C"], raise_error=True)
    with pytest.raises(ValueError):
        has_columns(test_input, ["B", "C"], exactly=True, raise_error=True)


def test_gather_columns():
    """Test gather_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert "B" not in gather_columns(test_input, ["A", "C"])


def test_to_range_index():
    """Test to_range_index fn."""
    test_df = pd.DataFrame(index=[1, 3], columns=["A", "B"])
    test_df.index.name = "index"
    test_df_with_range_index = to_range_index(test_df)
    test_df = pd.DataFrame(columns=["A", "B"])
    assert to_range_index(test_df).equals(test_df)
    assert has_range_index(test_df_with_range_index)


def test_is_timestamp_series():
    """Test is_timestamp_series fn."""
    assert is_timestamp_series(pd.to_datetime(["2000-01-21 01:30:00"]))
    with pytest.raises(ValueError):
        is_timestamp_series(pd.Series(["a", "b"]), raise_error=True)
