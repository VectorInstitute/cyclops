"""Test indexing util functions."""

import numpy as np
import pytest

from cyclops.utils.indexing import index_axis, take_indices, take_indices_over_axis


def test_index_axis():
    """Test index_axis fn."""
    indices = index_axis(3, 1, (10, 20, 30))
    assert indices[0] == slice(None, None, None)
    assert indices[1] == 3
    assert indices[2] == slice(None, None, None)

    indices = index_axis(4, 2, (10, 20, 30))
    assert indices[0] == slice(None, None, None)
    assert indices[0] == slice(None, None, None)
    assert indices[2] == 4


def test_take_indices():
    """Test take_indices fn."""
    arr = np.ones((3, 4, 5))
    arr[0, 2, 0] = 8
    indexed = take_indices(arr, (None, [2, 3], [0, 1]))
    assert indexed.shape == (3, 2, 2)
    assert indexed[0, 0, 0] == 8

    with pytest.raises(ValueError):
        take_indices(arr, ("cow", [2, 3], [0, 1]))


def test_take_indices_over_axis():
    """Test take_indices_over_axis fn."""
    arr = np.ones((3, 4, 5))
    arr[0, 2, 0] = 8
    indexed = take_indices_over_axis(arr, 1, [2])
    assert indexed.shape == (3, 1, 5)
    assert indexed[0, 0, 0] == 8
