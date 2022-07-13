"""Test vectorize.py."""

import numpy as np
import pytest

from cyclops.processors.feature.vectorize import Vectorized


@pytest.fixture
def input_data():
    """Input data."""
    return np.array(
        [
            [[1, 2, 3], [3, 4, 6]],
            [[3, 2, 1], [3, 2, 1]],
        ]
    ), [["0-0", "0-1"], ["1-0", "1-1"], ["2-0", "2-1", "2-2"]]


def test_vectorized(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test Vectorized."""
    data, indexes = input_data

    try:
        Vectorized(data, [["0-0", "0-1"], ["1-0", "1-1"], ["0-0", "1-1"]])
        raise ValueError(
            "Should have raised error where the last list must have 3 elements, not 2."
        )
    except ValueError:
        pass

    Vectorized(data, indexes)


def test_get_by_index(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test get_by_index function in Vectorized."""
    data, indexes = input_data

    vectorized = Vectorized(data, indexes)

    assert np.array_equal(
        vectorized.data, vectorized.get_by_index([None, None, None]).data
    )

    try:
        _ = vectorized.get_by_index([None, 1, None]).data
        raise ValueError("Should have raised error that the 1 element isn't a list")
    except ValueError:
        pass

    # Index and get back the axis
    expanded = np.expand_dims(data[:, 1, :], 1)
    assert np.array_equal(expanded, vectorized.get_by_index([None, [1], None]).data)


def test_get_by_value(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test get_by_value function in Vectorized."""
    data, indexes = input_data

    vectorized = Vectorized(data, indexes)

    assert np.array_equal(
        vectorized.data, vectorized.get_by_value([None, None, None]).data
    )

    try:
        _ = vectorized.get_by_value([None, "1-1", None]).data
        raise ValueError("Should have raised an error that the 1 element isn't a list")
    except ValueError:
        pass

    # Index and get back the axis
    expanded = np.expand_dims(data[:, 1, :], 1)
    assert np.array_equal(expanded, vectorized.get_by_value([None, ["1-1"], None]).data)
    assert np.array_equal(
        vectorized.get_by_value([None, ["1-1"], None]).data,
        vectorized.get_by_index([None, [1], None]).data,
    )
