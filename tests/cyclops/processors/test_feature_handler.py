"""Tests for feature handler module."""

import pytest

import numpy as np
import pandas as pd

from cyclops.processors.feature_handler import _category_to_numeric, FeatureHandler


def test_category_to_numeric():
    """Test _category_to_numeric fn."""
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
def test_input():
    """Create a test input."""
    input_ = pd.DataFrame(index=[0, 1, 2, 4], columns=["A", "B", "C"])
    input_.loc[0] = ["sheep", 10, "0"]
    input_.loc[1] = ["cat", 2, "0"]
    input_.loc[2] = ["cat", 3, "1"]
    input_.loc[4] = ["dog", 9.1, "0"]
    return input_


@pytest.fixture
def test_input_extra_column():
    """Create test input dataframe with single column to add."""
    input_ = pd.DataFrame(index=[0, 1, 2, 4], columns=["D"])
    input_.loc[0] = [15.0]
    input_.loc[1] = [5.1]
    return input_


def test_add_features(  # pylint: disable=redefined-outer-name
    test_input, test_input_extra_column
):
    """Test adding features."""
    feature_handler = FeatureHandler()
    feature_handler.add_features(test_input)
    assert (feature_handler.features["A-cat"].values == [0, 1, 1, 0]).all()
    assert (feature_handler.features["A-dog"].values == [0, 0, 0, 1]).all()
    assert (feature_handler.features["B"].values == [10, 2, 3, 9.1]).all()
    assert (feature_handler.features["C"].values == [0, 0, 1, 0]).all()

    feature_handler.add_features(test_input_extra_column)
    assert feature_handler.features["D"][0] == 15.0
    assert feature_handler.features["D"][1] == 5.1
    assert np.isnan(feature_handler.features["D"][2])
    assert np.isnan(feature_handler.features["D"][4])

    feature_handler = FeatureHandler(test_input)
    assert (feature_handler.features["A-cat"].values == [0, 1, 1, 0]).all()
    assert (feature_handler.features["A-dog"].values == [0, 0, 0, 1]).all()
    assert (feature_handler.features["B"].values == [10, 2, 3, 9.1]).all()
    assert (feature_handler.features["C"].values == [0, 0, 1, 0]).all()
