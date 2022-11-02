"""Test dataset splits."""

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.feature.split import (
    fractions_to_split,
    intersect_datasets,
    split_datasets,
    split_datasets_by_idx,
    split_idx,
)


def test_fractions_to_split():
    """Test fractions_to_split function."""
    assert fractions_to_split(0.8, 100) == [80]
    with pytest.raises(ValueError):
        fractions_to_split("donkey", 12)
    with pytest.raises(ValueError):
        fractions_to_split([0.8, 1], 12)
    with pytest.raises(ValueError):
        fractions_to_split([0.8, -0.2], 12)
    with pytest.raises(ValueError):
        fractions_to_split([0.8, 0.13, 0.23], 12)


def test_split_idx():
    """Test split_idx function."""
    split1 = split_idx([0.2, 0.8], 100, seed=3)
    split2 = split_idx(0.2, 100, seed=3)
    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[1], split2[1])


def test_split_datasets():
    """Test split_datasets function."""
    data = np.array([6, 3, 3, 54, 6, 3, 8, 6, 2, 1, 1, 9])
    labels = data.copy()
    _ = split_datasets(data, 0.8)
    _ = split_datasets(data, 0.8, axes=0)
    splits = split_datasets([data, labels], 0.8)
    train_data, val_data = splits[0]
    train_labels, val_labels = splits[1]
    assert np.array_equal(train_data, train_labels)
    assert np.array_equal(val_data, val_labels)

    labels = np.array([1])

    try:
        split_datasets([data, labels], 0.8)
        raise ValueError(
            (
                "An error should have been thrown since data/labels have a different",
                "number of samples along the axis.",
            )
        )
    except ValueError:
        pass


def test_intersect_datasets():
    """Test intersect_datasets fn."""
    dataframe1 = pd.DataFrame(
        [[1, 2, 3], [2, 3, 8], [3, 2, 0.2]], columns=["A", "B", "C"]
    )
    dataframe2 = pd.DataFrame(
        [[1, 4, 3], [4, 6.3, 8], [3, 2, 0.2]], columns=["A", "D", "E"]
    )
    datas = intersect_datasets([dataframe1, dataframe2], on_col="A")
    assert datas[0]["A"][0] == 1 and datas[0]["A"][2] == 3
    assert datas[1]["A"][0] == 1 and datas[1]["A"][2] == 3
    assert datas[0]["B"][0] == 2 and datas[0]["B"][2] == 2
    assert datas[0]["C"][0] == 3 and datas[0]["C"][2] == 0.2
    assert datas[1]["D"][0] == 4 and datas[1]["D"][2] == 2
    assert datas[1]["E"][0] == 3 and datas[1]["E"][2] == 0.2


def test_split_datasets_by_idx():
    """Test split_datasets_by_idx fn."""
    data = np.array([6, 3, 3, 54, 6, 3, 8, 6, 2, 1, 1, 9])
    splits = split_datasets_by_idx(
        data, (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])), axes=0
    )
    assert (splits[0] == np.array([6, 3, 6, 8])).all()
    assert (splits[1] == np.array([3, 54, 3, 6])).all()
