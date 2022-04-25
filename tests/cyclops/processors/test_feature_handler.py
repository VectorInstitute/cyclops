"""Tests for feature handler module."""

from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.feature_handler import FeatureHandler, _category_to_numeric


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
def test_input_static():
    """Create a test input (static)."""
    input_ = pd.DataFrame(index=[0, 1, 2, 4], columns=["A", "B", "C"])
    input_.loc[0] = ["sheep", 10, "0"]
    input_.loc[1] = ["cat", 2, "0"]
    input_.loc[2] = ["cat", 3, "1"]
    input_.loc[4] = ["dog", 9.1, "0"]

    return input_


@pytest.fixture
def test_input_static_extra_column():
    """Create test input dataframe with single column to add."""
    input_ = pd.DataFrame(index=[0, 1, 2, 4], columns=["D"])
    input_.loc[0] = [15.0]
    input_.loc[1] = [5.1]

    return input_


@pytest.fixture
def test_input_temporal_extra_column():
    """Create test input dataframe with single column to add."""
    index = pd.MultiIndex.from_tuples([("sheep", 0), ("cat", 0), ("cat", 1)])
    input_ = pd.DataFrame(index=index, columns=["D"])

    input_.loc[("sheep", 0)] = [0.7]
    input_.loc[("cat", 0)] = [0.8]
    input_.loc[("cat", 1)] = [1.9]

    return input_


@pytest.fixture
def test_input_temporal():
    """Create a test input (temporal)."""
    index = pd.MultiIndex.from_tuples(
        [("sheep", 0), ("cat", 0), ("cat", 1), ("dog", 0)]
    )
    input_ = pd.DataFrame(index=index, columns=["A", "B", "C"])
    input_.loc[("sheep", 0)] = ["cat", 10, "0"]
    input_.loc[("cat", 0)] = ["dog", 2, "0"]
    input_.loc[("cat", 1)] = ["cat", 3, "1"]
    input_.loc[("dog", 0)] = ["camel", 9.1, "0"]

    return input_


def test_add_features_temporal(  # pylint: disable=redefined-outer-name
    test_input_temporal, test_input_temporal_extra_column
):
    """Test adding features."""
    feature_handler = FeatureHandler()
    feature_handler.add_features(test_input_temporal)
    assert (feature_handler.temporal.loc[("sheep", 0)] == [0, 1, 0, 10.0, 0]).all()
    assert (feature_handler.temporal.loc[("cat", 0)] == [0, 0, 1, 2.0, 0]).all()
    assert (feature_handler.temporal.loc[("cat", 1)] == [0, 1, 0, 3.0, 1]).all()
    assert (feature_handler.temporal.loc[("dog", 0)] == [1, 0, 0, 9.1, 0]).all()

    feature_handler.add_features(test_input_temporal_extra_column)
    assert (feature_handler.temporal.loc[("sheep", 0)] == [0, 1, 0, 10.0, 0, 0.7]).all()
    assert (feature_handler.temporal.loc[("cat", 0)] == [0, 0, 1, 2.0, 0, 0.8]).all()
    assert (feature_handler.temporal.loc[("cat", 1)] == [0, 1, 0, 3.0, 1, 1.9]).all()


def test_add_features_static(  # pylint: disable=redefined-outer-name
    test_input_static, test_input_static_extra_column
):
    """Test adding features."""
    feature_handler = FeatureHandler()
    feature_handler.add_features(test_input_static)
    assert (feature_handler.static["A-cat"].values == [0, 1, 1, 0]).all()
    assert (feature_handler.static["A-dog"].values == [0, 0, 0, 1]).all()
    assert (feature_handler.static["B"].values == [10, 2, 3, 9.1]).all()
    assert (feature_handler.static["C"].values == [0, 0, 1, 0]).all()

    # Test properties.
    assert all(
        a == b
        for a, b in zip(feature_handler.names, ["A-cat", "A-dog", "A-sheep", "B", "C"])
    )
    TestCase().assertDictEqual(
        feature_handler.types,
        {
            "A-cat": "binary",
            "A-dog": "binary",
            "A-sheep": "binary",
            "B": "numeric",
            "C": "binary",
        },
    )

    feature_handler.add_features(test_input_static_extra_column)
    assert feature_handler.static["D"][0] == 15.0
    assert feature_handler.static["D"][1] == 5.1
    assert np.isnan(feature_handler.static["D"][2])
    assert np.isnan(feature_handler.static["D"][4])

    feature_handler = FeatureHandler(test_input_static)
    assert (feature_handler.static["A-cat"].values == [0, 1, 1, 0]).all()
    assert (feature_handler.static["A-dog"].values == [0, 0, 0, 1]).all()
    assert (feature_handler.static["B"].values == [10, 2, 3, 9.1]).all()
    assert (feature_handler.static["C"].values == [0, 0, 1, 0]).all()


def test_drop_features_static(  # pylint: disable=redefined-outer-name
    test_input_static,
):
    """Test dropping features."""
    feature_handler = FeatureHandler()
    feature_handler.add_features(test_input_static)
    assert (feature_handler.static["A-cat"].values == [0, 1, 1, 0]).all()
    assert (feature_handler.static["A-dog"].values == [0, 0, 0, 1]).all()
    assert (feature_handler.static["B"].values == [10, 2, 3, 9.1]).all()
    assert (feature_handler.static["C"].values == [0, 0, 1, 0]).all()

    # Drop single feature.
    feature_handler.drop_features(["B"])
    assert "B" not in feature_handler.static.columns

    # Drop categorical group of features.
    feature_handler.drop_features("A")
    assert all("A" not in col_name for col_name in feature_handler.static.columns)


def test_drop_features_temporal(  # pylint: disable=redefined-outer-name
    test_input_temporal,
):
    """Test dropping features."""
    feature_handler = FeatureHandler()
    feature_handler.add_features(test_input_temporal)
    assert (feature_handler.temporal.loc[("sheep", 0)] == [0, 1, 0, 10.0, 0]).all()
    assert (feature_handler.temporal.loc[("cat", 0)] == [0, 0, 1, 2.0, 0]).all()
    assert (feature_handler.temporal.loc[("cat", 1)] == [0, 1, 0, 3.0, 1]).all()
    assert (feature_handler.temporal.loc[("dog", 0)] == [1, 0, 0, 9.1, 0]).all()

    # Drop single feature.
    feature_handler.drop_features(["B"])
    assert "B" not in feature_handler.temporal.columns

    # Drop categorical group of features.
    feature_handler.drop_features("A")
    assert all("A" not in col_name for col_name in feature_handler.temporal.columns)
