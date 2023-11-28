"""Test dataset splits."""

import numpy as np
import pandas as pd
import pytest

from cyclops.data.df.split import (
    fractions_to_split,
    idxs_to_splits,
    intersect_datasets,
    kfold_cross_val,
    split_datasets,
    split_datasets_by_idx,
    split_idx,
    split_idx_stratified,
    split_kfold,
)


def test_fractions_to_split_results():
    """Test that the resulting index arrays are correct."""
    assert fractions_to_split(0.8, 100) == [80]
    assert np.array_equal(
        fractions_to_split([0.2] * 4, 100),
        np.array(range(20, 100, 20), dtype=int),
    )
    assert np.array_equal(fractions_to_split([1.0, 2.0], 12), np.array((4,), dtype=int))

    assert fractions_to_split([1.0], 12).size == 0
    assert fractions_to_split(1.0, 12).size == 0
    assert fractions_to_split([2.0], 12).size == 0

    # Expect no exception although sum > 1
    fractions_to_split([0.8, 0.13, 0.23], 12)


def test_fractions_to_split_expect_no_permutation():
    """Expect that no permutations to mutable sequence argument."""
    lst = [0.8, 0.2]
    fractions_to_split(lst, 100)
    assert lst == [0.8, 0.2]


@pytest.mark.parametrize(
    ("fractions", "n_samples", "expected_exception"),
    [
        (0.5, -10, ValueError),
        ("donkey", 12, TypeError),
        ([-0.01], 12, ValueError),
        (-0.01, 12, ValueError),
        ([0.8, "1"], 12, TypeError),
        ([0.8, complex(1, 2)], 12, TypeError),
        (complex(1, 0), 12, TypeError),
        ([0.8, -0.2], 12, ValueError),
        (-0.8, 12, ValueError),
    ],
)
def test_fractions_to_split_exceptions(fractions, n_samples, expected_exception):
    """Test expected exceptions in fractions_to_split function."""
    with pytest.raises(expected_exception):
        fractions_to_split(fractions, n_samples)


def test_split_idx():
    """Test split_idx function."""
    split1 = split_idx([0.2, 0.8], 100, seed=3)
    split2 = split_idx(0.2, 100, seed=3)
    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[1], split2[1])


def test_split_idx_stratified():
    """Test split_idx_stratified function."""
    stratify_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    idxs = split_idx_stratified(0.6, stratify_labels)
    _, val_counts1 = np.unique(stratify_labels[idxs[0]], return_counts=True)
    _, val_counts2 = np.unique(stratify_labels[idxs[1]], return_counts=True)
    assert np.array_equal(val_counts1, val_counts2)


def test_split_kfold():
    """Test split_kfold function."""
    splits = split_kfold(4, 100, seed=3)
    assert all(len(split) == 25 for split in splits)


def test_idxs_to_splits():
    """Test idxs_to_splits function."""
    splits = idxs_to_splits(
        np.array([1, 2, 100, 10]),
        (np.array([0, 2]), np.array([1, 3])),
    )
    assert np.array_equal(splits[0], np.array([1, 100]))
    assert np.array_equal(splits[1], np.array([2, 10]))


def test_kfold_cross_val():
    """Test kfold_cross_val function."""
    n_runs = 0
    for (
        train,
        val,
    ) in kfold_cross_val(5, np.array([1, 2, 3, 4, 5])):
        assert len(train) == 4
        assert len(val) == 1
        n_runs += 1

    assert n_runs == 5


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
            ),
        )
    except ValueError:
        pass


def test_intersect_datasets():
    """Test intersect_datasets fn."""
    dataframe1 = pd.DataFrame(
        [[1, 2, 3], [2, 3, 8], [3, 2, 0.2]],
        columns=["A", "B", "C"],
    )
    dataframe2 = pd.DataFrame(
        [[1, 4, 3], [4, 6.3, 8], [3, 2, 0.2]],
        columns=["A", "D", "E"],
    )
    datas = intersect_datasets([dataframe1, dataframe2], on_col="A")
    assert datas[0]["A"][0] == 1
    assert datas[0]["A"][2] == 3
    assert datas[1]["A"][0] == 1
    assert datas[1]["A"][2] == 3
    assert datas[0]["B"][0] == 2
    assert datas[0]["B"][2] == 2
    assert datas[0]["C"][0] == 3
    assert datas[0]["C"][2] == 0.2
    assert datas[1]["D"][0] == 4
    assert datas[1]["D"][2] == 2
    assert datas[1]["E"][0] == 3
    assert datas[1]["E"][2] == 0.2


def test_split_datasets_by_idx():
    """Test split_datasets_by_idx fn."""
    data = np.array([6, 3, 3, 54, 6, 3, 8, 6, 2, 1, 1, 9])
    splits = split_datasets_by_idx(
        data,
        (np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])),
        axes=0,
    )
    assert (splits[0] == np.array([6, 3, 6, 8])).all()
    assert (splits[1] == np.array([3, 54, 3, 6])).all()
