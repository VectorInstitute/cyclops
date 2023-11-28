"""Test processor utility functions."""


import numpy as np
import pandas as pd
import pytest

from cyclops.data.utils import (
    create_indicator_variables,
    gather_columns,
    has_columns,
    has_range_index,
    is_timestamp_series,
    to_range_index,
)


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
