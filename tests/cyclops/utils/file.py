"""Tests for utility file functions."""

import os
import shutil

import pandas as pd
import pytest

from cyclops.utils.file import save_dataframe


@pytest.fixture
def test_data():
    """Dummy dataframe for testing."""
    return pd.DataFrame([[1, "a", 1], [5.1, "b", 0]], columns=["col1", "col2", "col3"])


def test_save_dataframe(test_data):  # pylint: disable=redefined-outer-name
    """Test save fn."""
    save_dataframe(test_data, "test_save", "test_features")
    loaded_data = pd.read_parquet(os.path.join("test_save", "test_features.gzip"))
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")
    save_dataframe(test_data, "test_save", "test_features", prefix="donkey")
    loaded_data = pd.read_parquet(
        os.path.join("test_save", "doneky_test_features.gzip")
    )
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")

    # Does nothing, just prints log warning!
    save_dataframe("donkey", "test_save", "test_features")
