"""Tests for utility file functions."""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from cyclops.utils.file import (
    exchange_extension,
    load_array,
    load_dataframe,
    process_file_save_path,
    save_array,
    save_dataframe,
)


@pytest.fixture
def test_data_with_index():
    """Dummy dataframe for testing."""
    return pd.DataFrame(
        [[1, "a", 1], [5.1, "b", 0]], columns=["col1", "col2", "col3"], index=[1, 3]
    )


@pytest.fixture
def test_data_without_index():
    """Dummy dataframe for testing."""
    return pd.DataFrame([[1, "a", 1], [5.1, "b", 0]], columns=["col1", "col2", "col3"])


def test_save_dataframe(
    test_data_with_index, test_data_without_index
):  # pylint: disable=redefined-outer-name
    """Test save fn."""
    path = os.path.join("test_save", "test_features")
    save_dataframe(test_data_with_index, path)
    loaded_data = pd.read_parquet(path + ".parquet")
    assert loaded_data.equals(test_data_with_index)
    save_dataframe(test_data_without_index, path)
    loaded_data = pd.read_parquet(path + ".parquet")
    assert loaded_data.equals(test_data_without_index)

    with pytest.raises(ValueError):
        save_dataframe("donkey", "donkey")

    save_dataframe(test_data_with_index, path, file_format="csv")
    loaded_data = pd.read_csv(path + ".csv", index_col=[0])
    assert loaded_data.equals(test_data_with_index)

    save_dataframe(test_data_without_index, path, file_format="csv")
    loaded_data = pd.read_csv(path + ".csv", index_col=[0])
    assert loaded_data.equals(test_data_without_index)

    shutil.rmtree("test_save")

    with pytest.raises(ValueError):
        save_dataframe(test_data_with_index, path, file_format="donkey")


def test_exchange_extension():
    """Test exchange_extension fn."""
    assert exchange_extension("/tmp/file.txt", "csv") == "/tmp/file.csv"


def test_process_file_save_path():
    """Test process_file_save_path fn."""
    with pytest.raises(ValueError):
        process_file_save_path("/tmp/tmp2/file.txt", file_format="csv")

    process_file_save_path("/tmp/tmp2/file.txt", file_format="txt")
    assert os.path.isdir("/tmp/tmp2")
    shutil.rmtree("/tmp/tmp2")


def test_load_dataframe(
    test_data_with_index, test_data_without_index
):  # pylint: disable=redefined-outer-name
    """Test load_dataframe fn."""
    path = os.path.join("test_save", "test_features")
    save_dataframe(test_data_with_index, path)
    loaded_data = load_dataframe(path)
    assert loaded_data.equals(test_data_with_index)

    save_dataframe(test_data_with_index, path, file_format="csv")
    loaded_data = load_dataframe(path, file_format="csv")
    assert loaded_data.equals(test_data_with_index)

    save_dataframe(test_data_without_index, path, file_format="csv")
    loaded_data = load_dataframe(path, file_format="csv")
    assert loaded_data.equals(test_data_without_index)

    with pytest.raises(ValueError):
        load_dataframe(path, file_format="donkey")

    shutil.rmtree("test_save")


def test_save_array():
    """Test save_array fn."""
    path = os.path.join("test_save", "test_array.npy")
    save_array(np.array([1, 2]), path)
    assert os.path.isfile(path)

    path = os.path.join("test_save", "test_array.npz")
    with pytest.raises(ValueError):
        save_array(np.array([1, 2]), path, file_format="npz")

    with pytest.raises(ValueError):
        save_array("donkey", path, file_format="npz")

    shutil.rmtree("test_save")


def test_load_array():
    """Test load_array fn."""
    path = os.path.join("test_save", "test_array.npy")
    save_array(np.array([1, 2]), path)
    arr = load_array(path)
    assert (arr == np.array([1, 2])).all()

    with pytest.raises(ValueError):
        load_array(path, file_format="donkey")

    shutil.rmtree("test_save")
