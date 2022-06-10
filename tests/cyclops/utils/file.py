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
    path = os.path.join("test_save", "test_features")
    save_dataframe(test_data, path)
    loaded_data = pd.read_parquet(path)
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")
