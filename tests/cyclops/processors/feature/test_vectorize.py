"""Test vectorize.py."""

# pylint: disable=unbalanced-tuple-unpacking

import numpy as np
import pytest

from cyclops.processors.constants import STANDARD
from cyclops.processors.feature.vectorize import (
    Vectorized,
    intersect_vectorized,
    split_vectorized,
)


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

    with pytest.raises(ValueError):
        Vectorized(
            data, [["0-0", "0-1"], ["1-0", "1-1"], ["0-0", "1-1"]], ["A", "B", "C"]
        )
    with pytest.raises(ValueError):
        Vectorized(data, [["0-0", "0-1"], ["1-0", "1-1"]], ["A", "B", "C"])
    with pytest.raises(ValueError):
        Vectorized(
            data, ["0-0", ["1-0", "1-1"], ["2-0", "2-1", "2-2"]], ["A", "B", "C"]
        )
    with pytest.raises(ValueError):
        Vectorized(data, indexes, ["A", "B", 1])
    with pytest.raises(ValueError):
        Vectorized(
            data,
            [["0-0", "0-0"], ["1-0", "1-1"], ["2-0", "2-1", "2-2"]],
            ["A", "B", "C"],
        )
    with pytest.raises(ValueError):
        Vectorized("donkey", indexes, ["A", "B", "C"])
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])
    assert np.array_equal(vectorized.get_data(), data)


def test_take_with_indices(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test take_with_indices method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    # Index and get back the axis
    expanded = np.expand_dims(data[:, 1, :], 1)
    assert np.array_equal(expanded, vectorized.take_with_indices(1, [1]).data)


def test_take_with_index(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test take_with_index method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    # Index and get back the axis
    expanded = np.expand_dims(data[:, 1, :], 1)
    assert np.array_equal(expanded, vectorized.take_with_index(1, ["1-1"]).data)

    # Test that getting via indidces vs the index is the same
    assert np.array_equal(
        vectorized.take_with_index(1, ["1-1", "1-0"]).data,
        vectorized.take_with_indices(1, [1, 0]).data,
    )


def test_split_by_indices(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test split_by_indices method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    # Test a simple split
    vectorized0, vectorized1 = vectorized.split_by_indices(0, [[0], [1]])
    assert (vectorized.data[0, :, :] == vectorized0.data).all()
    assert (vectorized.data[1, :, :] == vectorized1.data).all()

    # Try a more complex split
    vectorized0, vectorized1 = vectorized.split_by_indices(2, [[2, 0], [1]])
    assert (
        np.stack([vectorized.data[:, :, 2], vectorized.data[:, :, 0]], axis=-1)
        == vectorized0.data
    ).all()
    assert (np.expand_dims(vectorized.data[:, :, 1], -1) == vectorized1.data).all()

    # Try a 3-split
    vectorized0, vectorized1, vectorized2 = vectorized.split_by_indices(
        2, [[2], [0], [1]]
    )
    assert (np.expand_dims(vectorized.data[:, :, 2], -1) == vectorized0.data).all()
    assert (np.expand_dims(vectorized.data[:, :, 0], -1) == vectorized1.data).all()
    assert (np.expand_dims(vectorized.data[:, :, 1], -1) == vectorized2.data).all()

    # Ensure the duplicate value error is raised
    try:
        vectorized0, vectorized1 = vectorized.split_by_indices(0, [[0, 1], [1]])
    except ValueError as error:
        assert "duplicate" in str(error).lower()

    # Test allowing drops
    # Allow - should work to drop index 1
    vectorized.split_by_indices(2, [[0], [2]], allow_drops=True)
    # Don't allow - shouldn't work to drop index 1
    try:
        vectorized.split_by_indices(2, [[0], [2]], allow_drops=False)
    except ValueError as error:
        assert "drop" in str(error).lower()


def test_split_by_index(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test split_by_index method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    vectorized0, vectorized1 = vectorized.split_by_index(2, [["2-2", "2-0"], ["2-1"]])
    assert (
        np.stack([vectorized.data[:, :, 2], vectorized.data[:, :, 0]], axis=-1)
        == vectorized0.data
    ).all()
    assert (np.expand_dims(vectorized.data[:, :, 1], -1) == vectorized1.data).all()

    # Test allowing drops
    # Allow - should work to drop index 0
    vectorized0, vectorized1 = vectorized.split_by_index(
        2, [["2-2"], ["2-1"]], allow_drops=True
    )

    # Don't allow - shouldn't work to drop index 0
    try:
        vectorized0, vectorized1 = vectorized.split_by_index(
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


def test_rename_axis(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test rename_axis method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    vectorized.rename_axis("B", "hello")
    assert vectorized.axis_names[1] == "hello"


def test_swap_axes(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test swap_axes method."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    shape = vectorized.shape
    index_lens = [len(index) for index in vectorized.indexes]
    axis_names = list(vectorized.axis_names)

    vectorized.swap_axes(0, 2)

    shape_after = vectorized.shape
    index_lens_after = [len(index) for index in vectorized.indexes]
    axis_names_after = vectorized.axis_names

    assert shape[0] == shape_after[2]
    assert shape[1] == shape_after[1]
    assert shape[2] == shape_after[0]
    assert index_lens[0] == index_lens_after[2]
    assert index_lens[1] == index_lens_after[1]
    assert index_lens[2] == index_lens_after[0]
    assert axis_names[0] == axis_names_after[2]
    assert axis_names[1] == axis_names_after[1]
    assert axis_names[2] == axis_names_after[0]


def test_intersect_vectorized(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test intersect_vectorized function."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    vec1, vec2 = intersect_vectorized([vectorized, vectorized], axes="B")

    assert (vec1.data == vec2.data).all()


def test_split_vectorized(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test split_vectorized function."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])

    vec_split1, vec_split2 = split_vectorized(
        [vectorized, vectorized], [0.8, 0.2], axes="C"
    )
    vec_split = split_vectorized([vectorized], [0.8, 0.2], axes="C", seed=4)
    split1_data, split2_data = vec_split[0]
    assert np.array_equal(
        split1_data.data, np.array([[[2, 1], [4, 3]], [[2, 3], [2, 3]]])
    )
    assert np.array_equal(split2_data.data, np.array([[[3], [6]], [[1], [1]]]))
    split1_data1, split1_data2 = vec_split1
    split2_data1, split2_data2 = vec_split2

    assert (split1_data1.data == split2_data1.data).all()
    assert (split1_data2.data == split2_data2.data).all()


def test_normalization(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test normalization."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])
    with pytest.raises(ValueError):
        vectorized.add_normalizer("B")
        vectorized.add_normalizer(
            "B", normalization_method=STANDARD, normalizer_map={"1-0": STANDARD}
        )
    vectorized.add_normalizer("B", normalization_method=STANDARD)
    vectorized.fit_normalizer()
    vectorized.normalize()

    for index_name in vectorized.get_index("B"):
        val_sum = np.nansum(vectorized.take_with_index("B", [index_name]).data)
        assert np.isclose(val_sum, 0)


def test_vectorized_normalizer_subset(  # pylint: disable=redefined-outer-name
    input_data,
):
    """Test VectorizedNormalizer subset method and related Vectorized handling."""
    data, indexes = input_data
    vectorized = Vectorized(data, indexes, ["A", "B", "C"])
    vectorized.add_normalizer("B", normalization_method=STANDARD)
    vectorized.fit_normalizer()

    # Split along the normalized axis, should affect the normalizers
    vec_in, vec_out = vectorized.split_out("B", ["1-0"])
    assert set(vec_in.normalizer.normalizer_map.keys()) == set(["1-1"])
    assert set(vec_in.normalizer.normalizers.keys()) == set(["1-1"])
    assert set(vec_out.normalizer.normalizer_map.keys()) == set(["1-0"])
    assert set(vec_out.normalizer.normalizers.keys()) == set(["1-0"])

    vec_in.normalize()
    vec_out.normalize()

    # Split along a different axis, should not affect the normalizers
    vec_in, vec_out = vectorized.split_out("A", ["0-0"])
    assert set(vec_in.normalizer.normalizer_map.keys()) == set(["1-0", "1-1"])
    assert set(vec_in.normalizer.normalizers.keys()) == set(["1-0", "1-1"])
    assert set(vec_out.normalizer.normalizer_map.keys()) == set(["1-0", "1-1"])
    assert set(vec_out.normalizer.normalizers.keys()) == set(["1-0", "1-1"])

    vec_in.normalize()
    vec_out.normalize()


def test_concat_over_axis():
    """Test Vectorized method concat_over_axis."""
    # Use this format to test the concatentation
    _ = """
    comb_vectorized = temp_vectorized.concat_over_axis(
        EVENT_NAME,
        tab_aggregated_vec.data,
        tab_aggregated_vec.get_index(EVENT_NAME)
    )

    _, save = comb_vectorized.split_out(
        EVENT_NAME,
        temp_vectorized.get_index(EVENT_NAME)
    )
    assert np.array_equal(save.data, temp_vectorized.data, equal_nan=True)

    _, save = comb_vectorized.split_out(
        EVENT_NAME, tab_aggregated_vec.get_index(EVENT_NAME)
    )
    assert np.array_equal(save.data, tab_aggregated_vec.data, equal_nan=True)
    """
