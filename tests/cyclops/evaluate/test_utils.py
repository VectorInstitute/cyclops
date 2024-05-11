"""Test utility functions for the evaluate package."""

import numpy as np
import pyarrow as pa
import pytest
from datasets import Dataset

from cyclops.evaluate.utils import check_required_columns, get_columns_as_array


def test_check_required_columns():
    """Test check_required_columns fn."""
    # Scenario 1: All required columns are present
    dataset_columns = ["name", "age", "email"]
    check_required_columns(dataset_columns, "name", ["age", "email"])
    check_required_columns(dataset_columns, "age")
    check_required_columns(dataset_columns, ["name"])

    # Scenario 2: Some required columns are missing
    with pytest.raises(ValueError) as excinfo:
        check_required_columns(dataset_columns, "name", "gender")
    assert "gender" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        check_required_columns(dataset_columns, ["name", "gender"])
    assert "gender" in str(excinfo.value)

    # Scenario 3: All required columns are missing
    with pytest.raises(ValueError) as excinfo:
        check_required_columns(dataset_columns, "height", "weight")
    assert "height" in str(excinfo.value) and "weight" in str(excinfo.value)

    # Scenario 4: Handling of None in required columns
    check_required_columns(dataset_columns, None, "name")
    check_required_columns(dataset_columns, "name", None)
    check_required_columns(dataset_columns, None)

    # Scenario 5: No required columns provided (should not raise an error)
    check_required_columns(dataset_columns)


def test_get_columns_as_array():
    """Test get_columns_as_array fn."""
    # Mock dataset creation
    data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    mock_dataset = Dataset.from_dict(data)

    # Scenario 1: Valid Dataset object with single column
    result = get_columns_as_array(mock_dataset, "column1")
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))  # Adjusted expectation

    # Scenario 2: Valid Dataset object with multiple columns
    result = get_columns_as_array(mock_dataset, ["column1", "column2"])
    np.testing.assert_array_equal(result, np.array([[1, 4], [2, 5], [3, 6]]))

    # Scenario 3: Valid pyarrow table with single column
    result = get_columns_as_array(pa.table(data), "column1")
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))  # Adjusted expectation

    # Scenario 4: Valid pyarrow table with multiple columns
    result = get_columns_as_array(pa.table(data), ["column1", "column2"])
    np.testing.assert_array_equal(result, np.array([[1, 4], [2, 5], [3, 6]]))

    # Scenario 5: Invalid dataset type
    with pytest.raises(TypeError):
        get_columns_as_array(["not", "a", "dataset"], "column1")
