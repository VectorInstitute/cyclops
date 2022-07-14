"""Test vectorize.py."""

# pylint: disable=unbalanced-tuple-unpacking

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
        Vectorized(
            data, [["0-0", "0-1"], ["1-0", "1-1"], ["0-0", "1-1"]], ["A", "B", "C"]
        )
        raise ValueError(
            "Should have raised error where the last list must have 3 elements, not 2."
        )
    except ValueError:
        pass

    Vectorized(data, indexes, ["A", "B", "C"])


def test_get_by_index(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test get_by_index method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

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
    """Test get_by_value method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

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


def test_split_by_index(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test split_by_index method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    # Test a simple split
    vectorized0, vectorized1 = vectorized.split_by_index(0, [[0], [1]])
    assert (vectorized.data[0, :, :] == vectorized0.data).all()
    assert (vectorized.data[1, :, :] == vectorized1.data).all()

    # Try a more complex split
    vectorized0, vectorized1 = vectorized.split_by_index(2, [[2, 0], [1]])
    assert (
        np.stack([vectorized.data[:, :, 2], vectorized.data[:, :, 0]], axis=-1)
        == vectorized0.data
    ).all()
    assert (np.expand_dims(vectorized.data[:, :, 1], -1) == vectorized1.data).all()

    # Try a 3-split
    vectorized0, vectorized1, vectorized2 = vectorized.split_by_index(
        2, [[2], [0], [1]]
    )
    assert (np.expand_dims(vectorized.data[:, :, 2], -1) == vectorized0.data).all()
    assert (np.expand_dims(vectorized.data[:, :, 0], -1) == vectorized1.data).all()
    assert (np.expand_dims(vectorized.data[:, :, 1], -1) == vectorized2.data).all()

    # Ensure the duplicate value error is raised
    try:
        vectorized0, vectorized1 = vectorized.split_by_index(0, [[0, 1], [1]])
    except ValueError as error:
        assert "duplicate" in str(error).lower()

    # Test allowing drops
    # Allow - should work to drop index 1
    vectorized.split_by_index(2, [[0], [2]], allow_drops=True)
    # Don't allow - shouldn't work to drop index 1
    try:
        vectorized.split_by_index(2, [[0], [2]], allow_drops=False)
    except ValueError as error:
        assert "drop" in str(error).lower()


def test_split_by_index_name(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test split_by_index_name method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    vectorized0, vectorized1 = vectorized.split_by_index_name(
        2, [["2-2", "2-0"], ["2-1"]]
    )
    assert (
        np.stack([vectorized.data[:, :, 2], vectorized.data[:, :, 0]], axis=-1)
        == vectorized0.data
    ).all()
    assert (np.expand_dims(vectorized.data[:, :, 1], -1) == vectorized1.data).all()

    # Test allowing drops
    # Allow - should work to drop index 0
    vectorized0, vectorized1 = vectorized.split_by_index_name(
        2, [["2-2"], ["2-1"]], allow_drops=True
    )

    # Don't allow - shouldn't work to drop index 0
    try:
        vectorized0, vectorized1 = vectorized.split_by_index_name(
            2, [["2-2"], ["2-1"]], allow_drops=False
        )
    except ValueError as error:
        assert "drop" in str(error).lower()


def test_split_by_fraction(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test split_by_fraction method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    vectorized0, vectorized1 = vectorized.split_by_fraction(0, 0.5, randomize=False)
    assert (vectorized.data[0, :, :] == vectorized0.data).all()
    assert (vectorized.data[1, :, :] == vectorized1.data).all()
