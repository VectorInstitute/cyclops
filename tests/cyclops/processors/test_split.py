"""Test dataset splits."""

import numpy as np
from cyclops.processors.split import (
    fractions_to_split,
    split_idx,
    split_data,
)

def test_fractions_to_split():
    """Test fractions_to_split function."""
    assert fractions_to_split(0.8, 100) == [80]

def test_split_idx():
    """Test split_idx function."""
    split1 = split_idx([0.2, 0.8], 100, seed=3)
    split2 = split_idx(0.2, 100, seed=3)
    assert np.array_equal(split1[0], split2[0])
    assert np.array_equal(split1[1], split2[1])

def test_split_data():
    """Test split_data function."""
    data = np.array([6, 3, 3, 54, 6, 3, 8, 6, 2, 1, 1, 9])
    labels = data.copy()
    splits = split_data([data, labels], 0.8)
    train_data, val_data = splits[0]
    train_labels, val_labels = splits[1]
    assert np.array_equal(train_data, train_labels)
    assert np.array_equal(val_data, val_labels)
    
    labels = np.array([1])
    
    try:
        split_data([data, labels], 0.8)
        raise ValueError(
            ("An error should have been thrown since data/labels have a different",
             "number of samples along the axis."
            )
        )
    except ValueError:
        pass