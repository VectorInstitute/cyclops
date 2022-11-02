"""Tests for utility file functions."""

import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from cyclops.utils.file import (
    concat_consequtive_dataframes,
    exchange_extension,
    join,
    listdir_nonhidden,
    load_array,
    load_dataframe,
    load_pickle,
    process_dir_save_path,
    process_file_save_path,
    save_array,
    save_consequtive_dataframes,
    save_dataframe,
    save_pickle,
    yield_dataframes,
    yield_pickled_files,
)


def test_yield_pickled_files():
    """Test yield_pickled_files fn."""
    os.makedirs("test_dir4", exist_ok=True)
    test_data = {"data": "some data"}
    save_pickle(test_data, "test_dir4/test_data1")
    save_pickle(test_data, "test_dir4/test_data2")
    save_pickle(test_data, "test_dir4/test_data3")
    for pkl_file_content in yield_pickled_files("test_dir4", skip_n=1):
        TestCase().assertDictEqual(test_data, pkl_file_content)
    shutil.rmtree("./test_dir4")


def test_concat_consequtive_save_dataframes():
    """Test concat_consequtive_dataframes and save_consequtive_dataframes fn."""
    test_df = pd.DataFrame([1, 2], columns=["a"])
    os.makedirs("test_dir2", exist_ok=True)
    save_dataframe(test_df, "test_dir2/df1")
    save_dataframe(test_df, "test_dir2/df2")
    save_dataframe(test_df, "test_dir2/df3")
    save_dataframe(test_df, "test_dir2/df4")
    for dataframe in concat_consequtive_dataframes("test_dir2", every_n=2):
        assert dataframe.equals(
            pd.DataFrame([1, 2, 1, 2], columns=["a"], index=[0, 1, 0, 1])
        )

    save_dataframe(test_df, "test_dir2/df5")
    count = 0
    for dataframe in concat_consequtive_dataframes("test_dir2", every_n=2):
        if count == 2:
            assert dataframe.equals(pd.DataFrame([1, 2], columns=["a"], index=[0, 1]))
        else:
            assert dataframe.equals(
                pd.DataFrame([1, 2, 1, 2], columns=["a"], index=[0, 1, 0, 1])
            )
        count += 1

    save_consequtive_dataframes("test_dir2", "test_dir3", every_n=2)
    df1 = load_dataframe("test_dir3/batch_0000.parquet")
    df2 = load_dataframe("test_dir3/batch_0001.parquet")
    assert df1.equals(pd.DataFrame([1, 2, 1, 2], columns=["a"], index=[0, 1, 0, 1]))
    assert df2.equals(pd.DataFrame([1, 2, 1, 2], columns=["a"], index=[0, 1, 0, 1]))

    shutil.rmtree("test_dir2")
    shutil.rmtree("test_dir3")


def test_yield_dataframes():
    """Test yield_dataframes fn."""
    test_df = pd.DataFrame([1, 2], columns=["a"])
    os.makedirs("test_dir1", exist_ok=True)
    save_dataframe(test_df, "test_dir1/df1")
    save_dataframe(test_df, "test_dir1/df2")
    save_dataframe(test_df, "test_dir1/df3")
    count = 0
    for dataframe in yield_dataframes("test_dir1", skip_n=1):
        assert dataframe.equals(test_df)
        count += 1
    assert count == 2
    shutil.rmtree("./test_dir1")


def test_join():
    """Test path join fn."""
    assert join("/mnt", "donkey") == "/mnt/donkey"
    assert join("/mnt", "donkey", "monkey") == "/mnt/donkey/monkey"
    assert join("\\Users", "monkey") == "/Users/monkey"


def test_process_dir_save_path():
    """Test process_dir_save_path fn."""
    tmp_dir_path = "./tmp_dir"
    assert process_dir_save_path(tmp_dir_path) == tmp_dir_path
    shutil.rmtree(tmp_dir_path)
    with pytest.raises(ValueError):
        process_dir_save_path("./tmp_dir", create_dir=False)
    with open("tmp_file", "w", encoding="utf8") as _:
        pass
    with pytest.raises(ValueError):
        process_dir_save_path("tmp_file")
    os.remove("tmp_file")
    os.makedirs(tmp_dir_path, exist_ok=True)
    assert process_dir_save_path(tmp_dir_path) == tmp_dir_path
    shutil.rmtree(tmp_dir_path)


def test_save_load_pickle():
    """Test save_pickle and load_pickle fn."""
    test_data = {"data": "some data"}
    save_path = save_pickle(test_data, "test_data")
    return_data = load_pickle(save_path)
    TestCase().assertDictEqual(test_data, return_data)

    test_data = [1, "a", 3.4]
    save_path = save_pickle(test_data, "test_data")
    return_data = load_pickle(save_path)
    TestCase().assertListEqual(test_data, return_data)

    os.remove("test_data.pkl")


def test_listdir_nonhidden():
    """Test listdir_nonhidden fn."""
    os.makedirs("./test_dir", exist_ok=True)
    with open("test_dir/tmp_file1", "w", encoding="utf8") as _:
        pass
    with open("test_dir/tmp_file2", "w", encoding="utf8") as _:
        pass
    with open("test_dir/.tmp_file1", "w", encoding="utf8") as _:
        pass
    non_hidden_files = listdir_nonhidden("test_dir")
    TestCase().assertListEqual(sorted(non_hidden_files), ["tmp_file1", "tmp_file2"])
    shutil.rmtree("./test_dir")


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

    path = os.path.join("test_save", "test_array.donkey")
    with pytest.raises(ValueError):
        load_array(path, file_format="donkey")

    shutil.rmtree("test_save")
