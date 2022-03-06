"""Tests for feature handler module."""

import pytest

import pandas as pd

from cyclops.processors.feature_handler import _category_to_numeric, FeatureHandler


def test_category_to_numeric():
    """Test category_to_numeric fn."""
    # Test case 1 - should be unchanged.
    input_series = pd.Series([1, 0, 0, 0, 1], index=[2, 4, 6, 8, 10])
    numeric_output = _category_to_numeric(input_series)
    assert (numeric_output == input_series).all()

    # Test case 2 - Convert string categories to indices.
    input_series = pd.Series(["A", "A", "B", "C", "B"])
    numeric_output = _category_to_numeric(input_series)
    assert (numeric_output == pd.Series([0, 0, 1, 2, 1])).all()

    # Test case 3 - Convert string categories using provided unique values,
    # leave others unchanged.
    input_series = pd.Series(["A", "A", "B", "C", "B"])
    numeric_output = _category_to_numeric(input_series, unique=["A", "B"])
    assert (numeric_output == pd.Series([0, 0, 1, "C", 1])).all()

    # Test case 4 - Test in-place replacement of series.
    input_series = pd.Series(["A", "A", "B", "C", "B"])
    numeric_output = _category_to_numeric(input_series, unique=["A", "B"], inplace=True)
    assert id(numeric_output) == id(input_series)
    numeric_output = _category_to_numeric(input_series, unique=["A", "B"])
    assert id(numeric_output) != id(input_series)


@pytest.fixture
def default_feature_handler():
    """Return default feature handler initialised with no input."""
    return FeatureHandler()


def test_add_features(default_feature_handler):
    test_input = pd.DataFrame(index=[0, 1, 2, 4], columns=["A", "B", "C"])
    test_input.loc[0] = ["sheep", 10, "0"]
    test_input.loc[1] = ["cat", 2, "0"]
    test_input.loc[2] = ["cat", 3, "1"]
    test_input.loc[4] = ["dog", 9.1, "0"]
    print(test_input)
    default_feature_handler.add_features(test_input)
    print(default_feature_handler.df, default_feature_handler.meta[0].group)
    assert 1 == 0
