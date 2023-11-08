"""Test utility functions for performing operations on Arrays."""
from collections import defaultdict, namedtuple

import numpy as np
import numpy.array_api as anp
import pytest
import torch

from cyclops.evaluate.metrics.experimental.utils.ops import (
    apply_to_array_collection,
    bincount,
    clone,
    dim_zero_cat,
    dim_zero_max,
    dim_zero_mean,
    dim_zero_min,
    dim_zero_sum,
    flatten,
    flatten_seq,
    moveaxis,
    remove_ignore_index,
    safe_divide,
    sigmoid,
    squeeze_all,
)
from cyclops.utils.optional import import_optional_module


cp = import_optional_module("cupy", error="ignore")


def multiply_by_two(x):
    """Multiply the input by two."""
    return x * 2


class TestApplyToArrayCollection:
    """Test the `apply_to_array_collection` utility function."""

    def test_apply_to_single_array(self):
        """Test applying a function to a single array."""
        data = anp.asarray([1, 2, 3, 4, 5])

        result = apply_to_array_collection(data, multiply_by_two)

        assert anp.all(result == anp.asarray([2, 4, 6, 8, 10]))

    def test_apply_to_list_of_arrays(self):
        """Test applying a function to a list of arrays."""
        data = [anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6])]

        result = apply_to_array_collection(data, multiply_by_two)
        expected_result = [anp.asarray([2, 4, 6]), anp.asarray([8, 10, 12])]

        assert all(anp.all(a == b) for a, b in zip(result, expected_result))

    def test_apply_to_tuple_of_arrays(self):
        """Test applying a function to a tuple of arrays."""
        data = (anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6]))

        result = apply_to_array_collection(data, multiply_by_two)
        expected_result = (anp.asarray([2, 4, 6]), anp.asarray([8, 10, 12]))

        assert all(anp.all(a == b) for a, b in zip(result, expected_result))

    def test_apply_to_dictionary_of_arrays(self):
        """Test applying a function to a dictionary of arrays."""
        data = {"a": anp.asarray([1, 2, 3]), "b": anp.asarray([4, 5, 6])}

        result = apply_to_array_collection(data, multiply_by_two)
        expected_result = {"a": anp.asarray([2, 4, 6]), "b": anp.asarray([8, 10, 12])}

        assert all(
            anp.all(a == b) for a, b in zip(result.values(), expected_result.values())
        )
        assert all(k in result for k in expected_result)

    def test_apply_to_namedtuple_of_arrays(self):
        """Test applying a function to a namedtuple of arrays."""
        Data = namedtuple("Data", ["a", "b"])
        data = Data(anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6]))

        result = apply_to_array_collection(data, multiply_by_two)
        expected_result = Data(anp.asarray([2, 4, 6]), anp.asarray([8, 10, 12]))

        assert all(anp.all(a == b) for a, b in zip(result, expected_result))
        assert all(k in result._fields for k in expected_result._fields)

    def test_return_input_data_if_not_array(self):
        """Test returning the input data for non-array inputs."""
        data = 10
        result = apply_to_array_collection(data, multiply_by_two)
        assert result == 10

    def test_return_input_data_if_not_array_collection(self):
        """Test returning the input data for non-array collections."""
        data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        result = apply_to_array_collection(data, multiply_by_two)
        assert result == data

    def test_handle_empty_list_input(self):
        """Test handling an empty list input."""
        result = apply_to_array_collection([], multiply_by_two)
        assert result == []

    def test_handle_empty_tuple_input(self):
        """Test handling an empty tuple input."""
        result = apply_to_array_collection((), multiply_by_two)
        assert result == ()

    def test_handle_empty_dictionary_input(self):
        """Test handling an empty dictionary input."""
        result = apply_to_array_collection({}, multiply_by_two)
        assert result == {}

    def test_handle_dictionary_with_non_string_keys(self):
        """Test handling a dictionary with non-string keys."""
        data = {1: anp.asarray([1, 2, 3]), 2: anp.asarray([4, 5, 6])}

        result = apply_to_array_collection(data, multiply_by_two)
        expected_result = {1: anp.asarray([2, 4, 6]), 2: anp.asarray([8, 10, 12])}
        assert all(
            anp.all(a == b) for a, b in zip(result.values(), expected_result.values())
        )
        assert all(k in result for k in expected_result)

    def test_handle_defaultdict_input(self):
        """Test handling a defaultdict input."""
        data = defaultdict(
            list,
            {"a": anp.asarray([1, 2, 3]), "b": anp.asarray([4, 5, 6])},
        )

        result = apply_to_array_collection(data, multiply_by_two)
        expected_result = defaultdict(
            list,
            {"a": anp.asarray([2, 4, 6]), "b": anp.asarray([8, 10, 12])},
        )

        assert all(
            anp.all(a == b) for a, b in zip(result.values(), expected_result.values())
        )
        assert all(k in result for k in expected_result)

    def test_apply_to_nested_collections(self):
        """Test applying a function to nested collections of arrays."""
        data = {
            "a": anp.asarray(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ],
            ),
            "b": [anp.asarray([10, 11, 12]), anp.asarray([13, 14, 15])],
            "c": (
                anp.asarray([16, 17, 18]),
                anp.asarray([19, 20, 21]),
            ),
            "d": {
                "e": anp.asarray([22, 23, 24]),
                "f": anp.asarray([25, 26, 27]),
            },
        }

        result = apply_to_array_collection(data, multiply_by_two)

        expected_result = {
            "a": anp.asarray(
                [
                    [2, 4, 6],
                    [8, 10, 12],
                    [14, 16, 18],
                ],
            ),
            "b": [anp.asarray([20, 22, 24]), anp.asarray([26, 28, 30])],
            "c": (
                anp.asarray([32, 34, 36]),
                anp.asarray([38, 40, 42]),
            ),
            "d": {
                "e": anp.asarray([44, 46, 48]),
                "f": anp.asarray([50, 52, 54]),
            },
        }

        for k in expected_result:
            assert k in result

            if isinstance(expected_result[k], dict):
                for kk in expected_result[k]:
                    assert kk in result[k]
                    assert anp.all(expected_result[k][kk] == result[k][kk])
            elif isinstance(expected_result[k], (tuple, list)):
                assert all(
                    anp.all(a == b) for a, b in zip(result[k], expected_result[k])
                )
            else:
                assert anp.all(expected_result[k] == result[k])


class TestBincount:
    """Test the `bincount` utility function."""

    def test_non_negative_integers(self):
        """Test using non-negative integers as input."""
        input_array = anp.asarray([0, 1, 1, 2, 2, 2])
        expected_output = anp.asarray([1, 2, 3])

        result = bincount(input_array)

        assert anp.all(result == expected_output)

    def test_empty_array(self):
        """Test using an empty array as input."""
        input_array = anp.asarray([], dtype=anp.int32)
        expected_output = anp.asarray([], dtype=anp.int64)

        result = bincount(input_array, minlength=0)

        assert anp.all(result == expected_output)

        result = bincount(input_array, minlength=10)
        expected_output = anp.zeros(10, dtype=anp.int64)

        assert anp.all(result == expected_output)

    def test_single_unique_value(self):
        """Test using an array with a single unique value as input."""
        input_array = anp.asarray([3, 3, 3, 3])
        expected_output = anp.asarray([0, 0, 0, 4])

        result = bincount(input_array)

        assert anp.all(result == expected_output)

    def test_no_repeated_values(self):
        """Test using an array with no repeated values as input."""
        input_array = anp.asarray([0, 1, 2, 3, 4, 5])
        expected_output = anp.ones_like(input_array)

        result = bincount(input_array)

        assert anp.all(result == expected_output)

    def test_negative_integers(self):
        """Test using an array with negative integers as input."""
        input_array = anp.asarray([-1, 0, 1, 2])

        with pytest.raises(ValueError):
            bincount(input_array)

    def test_negative_minlength(self):
        """Test using a negative minlength as input."""
        input_array = anp.asarray([1, 2, 3])

        with pytest.raises(ValueError):
            bincount(input_array, minlength=-5)

    def test_different_shapes(self):
        """Test using arrays and weights with different shapes as input."""
        input_array = anp.asarray([1, 2, 3])
        weights = anp.asarray([0.5, 0.5])

        with pytest.raises(ValueError):
            bincount(input_array, weights=weights)

    def test_not_one_dimensional(self):
        """Test using a multi-dimensional array as input."""
        input_array = anp.asarray([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            bincount(input_array)

    def test_not_integer_type(self):
        """Test using a non-integer array as input."""
        input_array = anp.asarray([1.5, 2.5, 3.5])

        with pytest.raises(ValueError):
            bincount(input_array)


class TestClone:
    """Test the `clone` utility function."""

    def test_clone_numpy_array(self):
        """Test if the clone function creates a new copy of a numpy array."""
        x = np.array([1, 2, 3])

        y = clone(x)

        # Check if y is a new copy of x
        assert y is not x
        assert np.array_equal(y, x)

    @pytest.mark.skipif(cp is None, reason="Cupy is not installed.")
    @pytest.mark.integration_test()  # machine for integration test has GPU
    def test_clone_cupy_array(self):
        """Test if the clone function creates a new copy of a cupy array."""
        try:
            if not cp.cuda.is_available():  # type: ignore
                pytest.skip("CUDA is not available.")
        except cp.cuda.runtime.CUDARuntimeError:  # type: ignore
            pytest.skip("CUDA is not available.")

        x = cp.asarray([1, 2, 3])  # type: ignore

        y = clone(x)

        # Check if y is a new copy of x
        assert y is not x
        assert cp.array_equal(y, x)  # type: ignore

    def test_clone_torch_tensor(self):
        """Test if the clone function properly clones a torch tensor."""
        x = torch.tensor([1, 2, 3])  # type: ignore

        y = clone(x)

        # Check if y is a new copy of x
        assert y is not x
        assert torch.equal(y, x)  # type: ignore

    def test_clone_empty_array(self):
        """Test if the clone function creates a new copy of an empty array."""
        x = anp.asarray([])

        y = clone(x)

        # Check if y is a new copy of x
        assert y is not x
        assert anp.all(y == x)


class TestDimZeroCat:
    """Test the `dim_zero_cat` utility function."""

    def test_returns_input_if_array_or_empty_list_tuple(self):
        """Test if the input is an array or empty list/tuple."""
        array1 = anp.asarray([1, 2, 3])
        array2 = anp.asarray([4, 5, 6])
        empty_list = []

        result1 = dim_zero_cat(array1)
        result2 = dim_zero_cat([array2])
        result3 = dim_zero_cat([])
        result4 = dim_zero_cat(empty_list)

        np.testing.assert_array_equal(result1, array1)
        np.testing.assert_array_equal(result2, array2)
        assert result3 == []
        assert result4 == []

    def test_concatenates_arrays_along_zero_dimension(self):
        """Test concatenation along the zero dimension."""
        array1 = anp.asarray([1, 2, 3])
        array2 = anp.asarray([4, 5, 6])

        result = dim_zero_cat([array1, array2])

        expected_result = anp.asarray([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result, expected_result)

    def test_arrays_with_different_shapes(self):
        """Test handling of arrays with different shapes."""
        array1 = anp.asarray([1, 2, 3])
        array2 = anp.asarray([[4, 5, 6], [7, 8, 9]])

        with pytest.raises(ValueError):
            dim_zero_cat([array1, array2])

    def test_raises_type_error_if_input_not_array_or_list_tuple_of_arrays(self):
        """Test raising TypeError if input is not an array or a list/tuple of arrays."""
        array1 = anp.asarray([1, 2, 3])
        array2 = anp.asarray([4, 5, 6])

        with pytest.raises(TypeError):
            dim_zero_cat([array1, array2, 7])

        with pytest.raises(TypeError):
            dim_zero_cat([array1, array2, None, "hello"])

        with pytest.raises(TypeError):
            dim_zero_cat(123)

    def test_raises_value_error_if_input_list_empty(self):
        """Test raising ValueError if input list is empty."""
        result = dim_zero_cat([])
        assert result == []


def test_dim_zero_max():
    """Test the `dim_zero_max` utility function."""
    # happy path
    array1 = anp.asarray([1, 2, 3])  # 1d
    array2 = anp.asarray([[4, 5, 6], [7, 8, 9]])

    result1 = dim_zero_max(array1)
    result2 = dim_zero_max(array2)

    expected_result1 = anp.asarray(3)
    np.testing.assert_array_equal(result1, expected_result1)
    expected_result2 = anp.asarray([7, 8, 9])
    np.testing.assert_array_equal(result2, expected_result2)

    # edge cases
    with pytest.raises(ValueError):
        dim_zero_max(anp.asarray([]))

    with pytest.raises(AttributeError):
        dim_zero_max([array1, array2])

    with pytest.raises(TypeError):
        dim_zero_max([1, 2, 3])

    # 1x1x1 array
    array3 = anp.asarray([[[1]]])
    result4 = dim_zero_max(array3)
    expected_result4 = anp.asarray([[1]])
    np.testing.assert_array_equal(result4, expected_result4)


def test_dim_zero_mean():
    """Test the `dim_zero_mean` utility function."""
    # happy path
    array1 = anp.asarray([1, 2, 3])
    array2 = anp.asarray([[4, 5, 6], [7, 8, 9]])
    array3 = anp.asarray([[[10, 11, 12], [13, 14, 15]], [[16, 17, 18], [19, 20, 21]]])

    result1 = dim_zero_mean(array1)
    expected_result1 = anp.asarray(2)
    np.testing.assert_array_equal(result1, expected_result1)

    result2 = dim_zero_mean(array2)
    expected_result2 = anp.asarray([5.5, 6.5, 7.5])
    np.testing.assert_allclose(result2, expected_result2)

    result3 = dim_zero_mean(array3)
    expected_result3 = anp.asarray([[13, 14, 15], [16, 17, 18]], dtype=anp.float32)
    np.testing.assert_allclose(result3, expected_result3)

    # edge cases
    result4 = dim_zero_mean(anp.asarray([]))
    np.testing.assert_array_equal(result4, anp.asarray(anp.nan))

    with pytest.raises(AttributeError):
        dim_zero_mean([array1, array2])

    with pytest.raises(TypeError):
        dim_zero_mean([1, 2, 3])


def test_dim_zero_min():
    """Test the `dim_zero_min` utility function."""
    # expected behavior
    array1 = anp.asarray([1, 2, 3])
    array2 = anp.asarray([[4, 5, 6], [7, 8, 9]])
    array3 = anp.asarray([[[10, 11, 12], [13, 14, 15]], [[-16, 17, 18], [19, 20, 21]]])

    result1 = dim_zero_min(array1)
    expected_result1 = anp.asarray(1)
    np.testing.assert_array_equal(result1, expected_result1)

    result2 = dim_zero_min(array2)
    expected_result2 = anp.asarray([4, 5, 6])
    np.testing.assert_array_equal(result2, expected_result2)

    result3 = dim_zero_min(array3)
    expected_result3 = anp.asarray([[-16, 11, 12], [13, 14, 15]])
    np.testing.assert_array_equal(result3, expected_result3)

    # edge cases
    with pytest.raises(ValueError):
        dim_zero_min(anp.asarray([]))

    with pytest.raises(AttributeError):
        dim_zero_min([array1, array2])

    with pytest.raises(TypeError):
        dim_zero_min([1, 2, 3])


def test_dim_zero_sum():
    """Test the `dim_zero_sum` utility function."""
    array1 = anp.asarray([1, 2, 3])
    array2 = anp.asarray([[4, 5, 6], [7, 8, 9]])
    array3 = anp.asarray([[[10, 11, 12], [13, 14, 15]], [[16, 17, 18], [19, 20, 21]]])

    result1 = dim_zero_sum(array1)
    expected_result1 = anp.asarray(6)
    np.testing.assert_array_equal(result1, expected_result1)

    result2 = dim_zero_sum(array2)
    expected_result2 = anp.asarray([11, 13, 15])
    np.testing.assert_array_equal(result2, expected_result2)

    result3 = dim_zero_sum(array3)
    expected_result3 = anp.asarray([[26, 28, 30], [32, 34, 36]])
    np.testing.assert_array_equal(result3, expected_result3)

    # edge cases
    result4 = dim_zero_sum(anp.asarray([]))
    np.testing.assert_array_equal(result4, anp.asarray(0))

    with pytest.raises(AttributeError):
        dim_zero_sum([array1, array2])

    with pytest.raises(TypeError):
        dim_zero_sum([1, 2, 3])


def test_flatten():
    """Test the `flatten` utility function."""
    x = anp.asarray([1, 2, 3])
    result = flatten(x)
    assert anp.all(result == x)
    assert not np.shares_memory(result, x)

    x = anp.asarray([[1, 2, 3], [4, 5, 6]])
    result = flatten(x)
    expected_result = anp.asarray([1, 2, 3, 4, 5, 6])
    assert anp.all(result == expected_result)
    assert not np.shares_memory(result, x)

    x = anp.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = flatten(x)
    expected_result = anp.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    assert anp.all(result == expected_result)
    assert not np.shares_memory(result, x)

    x = anp.asarray([])
    result = flatten(x)
    assert anp.all(result == x)
    assert not np.shares_memory(result, x)


def test_flatten_seq():
    """Test the `flatten_seq` utility function."""
    # happy path
    x = []
    assert flatten_seq(x) == []

    x = [1, 2, 3]
    assert flatten_seq(x) == x

    x = [[1, 2, 3], [4, 5, 6]]
    assert flatten_seq(x) == [1, 2, 3, 4, 5, 6]

    x = [[1, [2, [3]]], [4, [5, [6]]]]
    assert flatten_seq(x) == [1, 2, 3, 4, 5, 6]

    x = [[1, 2, 3], "abc", [4, [5, 6]], 7, anp.asarray(1)]
    assert flatten_seq(x) == [
        1,
        2,
        3,
        "a",
        "b",
        "c",
        4,
        5,
        6,
        7,
        anp.asarray(1),
    ]

    # edge cases
    x = 123
    with pytest.raises(TypeError):
        flatten_seq(x)  # type: ignore

    x = [1, None, [2, None, 3]]
    assert flatten_seq(x) == [1, None, 2, None, 3]

    x = [[], [1, 2], [], [3, 4], []]
    assert flatten_seq(x) == [1, 2, 3, 4]

    x = [[None, None], [None, None]]
    assert flatten_seq(x) == [None, None, None, None]

    x = [[], [], []]
    assert flatten_seq(x) == []


class TestMoveAxis:
    """Test the `moveaxis` utility function."""

    def test_move_single_axis(self):
        """Test moving a single axis."""
        array = anp.zeros((2, 3, 4))

        result = moveaxis(array, 0, 2)
        expected_result = np.moveaxis(array, 0, 2)  # type: ignore

        assert result.shape == expected_result.shape
        assert np.shares_memory(result, array)
        assert np.all(result == expected_result)

    def test_move_negative_indices(self):
        """Test moving an axis with negative indices."""
        array = anp.zeros((2, 3, 4))

        result = moveaxis(array, -1, -3)
        expected_result = np.moveaxis(array, -1, -3)  # type: ignore

        assert result.shape == expected_result.shape
        assert np.shares_memory(result, array)

        assert np.all(result == expected_result)

    def test_move_multiple_axes(self):
        """Test moving multiple axes."""
        array = anp.zeros((2, 3, 4))

        result = moveaxis(array, (0, 1), (1, 0))  # type: ignore
        expected_result = np.moveaxis(array, (0, 1), (1, 0))  # type: ignore

        assert result.shape == expected_result.shape
        assert np.shares_memory(result, array)
        assert np.all(result == expected_result)

    def test_move_same_position(self):
        """Test moving an axis to the same position."""
        array = anp.zeros((2, 3, 4))

        result = moveaxis(array, 0, 0)

        assert result.shape == (2, 3, 4)
        assert np.shares_memory(result, array)
        assert np.all(result == array)

    def test_move_outside_shape(self):
        """Test moving an axis outside the shape of the array."""
        array = anp.zeros((2, 3, 4))

        with pytest.raises(ValueError):
            moveaxis(array, 0, 5)

        with pytest.raises(ValueError):
            moveaxis(array, 0, -5)

    def test_raise_value_error_if_duplicate_values(self):
        """Test passing duplicate values for source or destination."""
        array = anp.zeros((2, 3, 4))

        with pytest.raises(ValueError):
            moveaxis(array, (0, 0), (1, 2))  # type: ignore

        with pytest.raises(ValueError):
            moveaxis(array, (0, 1), (1, 1))  # type: ignore

    def test_raise_value_error_if_source_and_destination_not_same_length(self):
        """Test passing source and destination with different lengths."""
        array = anp.zeros((2, 3, 4))

        with pytest.raises(ValueError):
            moveaxis(array, (0, 1), (1, 0, 2))  # type: ignore

    def test_raise_value_error_if_source_or_destination_not_integer_or_tuple(self):
        """Test passing source or destination as a non-integer or non-tuple."""
        array = anp.zeros((2, 3, 4))

        # Test with source as a float
        with pytest.raises(ValueError):
            moveaxis(array, 0.5, 2)  # type: ignore

        # Test with destination as a string
        with pytest.raises(ValueError):
            moveaxis(array, 0, "2")  # type: ignore

        # Test with source as a list
        with pytest.raises(ValueError):
            moveaxis(array, [0], 2)  # type: ignore

        # Test with destination as a dictionary
        with pytest.raises(ValueError):
            moveaxis(array, 0, {"2": 2})  # type: ignore


class TestRemoveIgnoreIndex:
    """Test the `remove_ignore_index` utility function."""

    def test_return_same_input_arrays_if_ignore_index_is_none(self):
        """Test case when `ignore_index` is None."""
        input_arrays = (anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6]))
        ignore_index = None

        result = remove_ignore_index(*input_arrays, ignore_index=ignore_index)

        assert result == input_arrays

    def test_remove_samples_equal_to_ignore_index_from_input_arrays(self):
        """Test removing samples that are equal to `ignore_index`."""
        target = anp.asarray([1, 2, 3])
        preds = anp.asarray([4, 5, 6])
        ignore_index = 2
        expected_target = anp.asarray([1, 3])
        expected_preds = anp.asarray([4, 6])

        out_target, out_preds = remove_ignore_index(
            target,
            preds,
            ignore_index=ignore_index,
        )

        assert anp.all(out_target == expected_target)
        assert anp.all(out_preds == expected_preds)

        target = anp.asarray([[1, 2, 3], [4, 5, 6]])
        preds = anp.asarray([[7, 8, 9], [10, 11, 12]])
        ignore_index = 1
        expected_target = anp.asarray([2, 3, 4, 5, 6])
        expected_preds = anp.asarray([8, 9, 10, 11, 12])

        out_target, out_preds = remove_ignore_index(
            target,
            preds,
            ignore_index=ignore_index,
        )

        assert anp.all(out_target == expected_target)
        assert anp.all(out_preds == expected_preds)

    def test_return_same_output_arrays_if_ignore_index_not_in_input_arrays(self):
        """Test returning the same arrays if `ignore_index` is not in array."""
        input_arrays = (anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6]))
        ignore_index = 7

        result = remove_ignore_index(*input_arrays, ignore_index=ignore_index)

        assert all(anp.all(a == b) for a, b in zip(result, input_arrays))

    def test_raise_type_error_if_ignore_index_not_integer_or_tuple_of_integers(self):
        """Test raising TypeError on invalid `ignore_index` type."""
        input_arrays = (anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6]))
        ignore_index = "ignore"

        with pytest.raises(TypeError):
            remove_ignore_index(*input_arrays, ignore_index=ignore_index)  # type: ignore

    def test_raise_type_error_if_input_arrays_not_array_objects(self):
        """Test raising TypeError on invalid input array type."""
        input_arrays = ([1, 2, 3], [4, 5, 6])
        ignore_index = 2

        with pytest.raises(TypeError):
            remove_ignore_index(*input_arrays, ignore_index=ignore_index)

    def test_return_empty_tuple_if_all_input_arrays_empty(self):
        """Test with all input arrays empty."""
        input_arrays = (anp.asarray([]), anp.asarray([]))
        ignore_index = 2

        result = remove_ignore_index(*input_arrays, ignore_index=ignore_index)

        assert all(anp.all(a == b) for a, b in zip(result, input_arrays))

    def test_return_empty_tuple_if_all_samples_are_equal_to_ignore_index(self):
        """Test ignoring all samples in input arrays."""
        input_arrays = (anp.asarray([1, 1, 1]), anp.asarray([1, 1, 1]))
        ignore_index = 1
        expected_result = (
            anp.asarray([], dtype=anp.int64),
            anp.asarray([], dtype=anp.int64),
        )

        result = remove_ignore_index(*input_arrays, ignore_index=ignore_index)

        assert all(anp.all(a == b) for a, b in zip(result, expected_result))

    def test_remove_samples_with_tuple_ignore_index(self):
        """Test with tuple of ignore_index values."""
        input_arrays = (anp.asarray([1, 2, 3]), anp.asarray([4, 5, 6]))
        ignore_index = (2, 3)

        result = remove_ignore_index(*input_arrays, ignore_index=ignore_index)

        expected_result = (anp.asarray([1]), anp.asarray([4]))
        assert all(anp.all(a == b) for a, b in zip(result, expected_result))

        input_arrays = (
            anp.asarray([[1, 2, 3], [4, 5, 6]]),
            anp.asarray([[7, 8, 9], [10, 11, 12]]),
        )
        ignore_index = (2, 6)

        result = remove_ignore_index(*input_arrays, ignore_index=ignore_index)

        expected_result = (anp.asarray([1, 3, 4, 5]), anp.asarray([7, 9, 10, 11]))
        assert all(anp.all(a == b) for a, b in zip(result, expected_result))


class TestSafeDivide:
    """Test the `safe_divide` utility function."""

    def test_divide_non_zero_denominators(self):
        """Test dividing two arrays with non-zero denominators."""
        numerator = anp.asarray([1.0, 2.0, 3.0])
        denominator = anp.asarray([2.0, 3.0, 4.0])
        expected_result = anp.asarray([0.5, 0.66666667, 0.75])

        result = safe_divide(numerator, denominator)

        assert np.allclose(result, expected_result)

    def test_divide_zero_denominators(self):
        """Test dividing two arrays with zero denominators, return array of zeros."""
        numerator = anp.asarray([1.0, 2.0, 3.0])
        denominator = anp.asarray([0.0, 0.0, 0.0])
        expected_result = anp.asarray([0.0, 0.0, 0.0])

        result = safe_divide(numerator, denominator)

        assert anp.all(result == expected_result)

    def test_divide_one_zero_denominator(self):
        """Test dividing two arrays with one zero denominator."""
        numerator = anp.asarray([1.0, 2.0, 3.0])
        denominator = anp.asarray([2.0, 0.0, 4.0])
        expected_result = anp.asarray([0.5, 0.0, 0.75])

        result = safe_divide(numerator, denominator)

        assert anp.all(result == expected_result)

    def test_divide_empty_arrays(self):
        """Test dividing two empty arrays."""
        import numpy.array_api as anp

        numerator = anp.asarray([])
        denominator = anp.asarray([])
        expected_result = anp.asarray([])

        result = safe_divide(numerator, denominator)

        assert anp.all(result == expected_result)

    def test_divide_different_datatypes(self):
        """Test dividing two arrays with different datatypes."""
        numerator = anp.asarray([1.0, 2.0, 3.0], dtype=anp.float32)
        denominator = anp.asarray([2.0, 3.0, 4.0], dtype=anp.float64)
        expected_result = anp.asarray([0.5, 0.66666667, 0.75], dtype=anp.float64)

        result = safe_divide(numerator, denominator)

        assert np.allclose(result, expected_result)

    def test_divide_mixed_values(self):
        """Test dividing two arrays with mixed positive and negative values."""
        numerator = anp.asarray([1.0, -2.0, 0.0, 3.0])
        denominator = anp.asarray([-2.0, 3.0, 0.0, -4.0])
        expected_result = anp.asarray([-0.5, -0.66666667, 0.0, -0.75])

        result = safe_divide(numerator, denominator)

        assert np.allclose(result, expected_result)

    def test_divide_large_values(self):
        """Test dividing two arrays with large values."""
        numerator = anp.asarray([1e20, 2e20, 3e20], dtype=anp.float64)
        denominator = anp.asarray([1e20, 1e-20, 3e20], dtype=anp.float32)
        expected_result = anp.asarray([1.0, 2e40, 1.0])

        result = safe_divide(numerator, denominator)

        assert np.allclose(result, expected_result)

    def test_divide_different_shapes(self):
        """Test dividing two arrays with different shapes."""
        numerator = anp.asarray([1.0, 2.0, 3.0])
        denominator = anp.asarray([1.0, 2.0])

        with pytest.raises(ValueError):
            safe_divide(numerator, denominator)

    def test_divide_inf_values(self):
        """Test dividing two arrays with Inf values."""
        numerator = np.asarray([1.0, 2.0, np.inf])
        denominator = np.asarray([2.0, np.inf, 4.0])
        expected_result = np.asarray([0.5, 0.0, np.inf])

        result = safe_divide(numerator, denominator)
        print(result)

        assert np.all(result == expected_result)

    def test_divide_with_nan_values(self):
        """Test dividing two arrays with NaN values."""
        numerator = np.asarray([1.0, 2.0, np.nan])
        denominator = np.asarray([2.0, np.nan, 4.0])
        expected_result = np.asarray([0.5, np.nan, np.nan])

        result = safe_divide(numerator, denominator)

        assert np.all(np.isnan(result) == np.isnan(expected_result))

    def test_returns_array_with_same_shape(self):
        """Test that the shape of the output is the same as the input arrays."""
        numerator = anp.asarray([1.0, 2.0, 3.0])
        denominator = anp.asarray([2.0, 3.0, 4.0])

        result = safe_divide(numerator, denominator)

        assert result.shape == numerator.shape
        assert result.shape == denominator.shape


class TestSigmoid:
    """Test the `sigmoid` utility function."""

    def test_sigmoid_positive_values(self):
        """Test sigmoid function with positive values."""
        x = anp.asarray([1.1, 2.0, 3.0])
        result = sigmoid(x)
        expected = anp.asarray([0.75026011, 0.88079708, 0.95257413], dtype=anp.float64)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_negative_values(self):
        """Test sigmoid function with negative values."""
        x = anp.asarray([-1.1, -2.0, -3.0])
        result = sigmoid(x)
        expected = anp.asarray([0.24973989, 0.11920292, 0.04742587], dtype=anp.float64)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_zeros(self):
        """Test sigmoid function with zeros."""
        x = anp.asarray([0, 0, 0])
        result = sigmoid(x)
        expected = anp.asarray([0.5, 0.5, 0.5], dtype=anp.float64)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_large_values(self):
        """Test sigmoid function with large values."""
        x = anp.asarray([100, 1000, 10000])
        result = sigmoid(x)
        expected = anp.asarray([1.0, 1.0, 1.0], dtype=anp.float64)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_small_values(self):
        """Test sigmoid function with small values."""
        x = anp.asarray([-100, -1000, -10000], dtype=anp.float32)
        result = sigmoid(x)
        expected = anp.asarray([0.0, 0.0, 0.0], dtype=anp.float32)
        np.testing.assert_allclose(result, expected, atol=4e-44)

    def test_sigmoid_empty_array(self):
        """Test sigmoid function with empty array."""
        x = anp.asarray([])
        result = sigmoid(x)
        expected = anp.asarray([], dtype=anp.float64)
        assert all(result == expected)

    def test_sigmoid_nan_values(self):
        """Test sigmoid function with NaN values."""
        x = anp.asarray([anp.nan, anp.nan, anp.nan])
        result = sigmoid(x)
        expected = anp.asarray([anp.nan, anp.nan, anp.nan], dtype=anp.float64)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_sigmoid_infinity_values(self):
        """Test sigmoid function with infinity values."""
        x = anp.asarray([anp.inf, anp.inf, anp.inf])
        result = sigmoid(x)
        expected = anp.asarray([1.0, 1.0, 1.0], dtype=anp.float64)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_negative_infinity_values(self):
        """Test sigmoid function with negative infinity values."""
        x = anp.asarray([-anp.inf, -anp.inf, -anp.inf])
        result = sigmoid(x)
        expected = anp.asarray([0.0, 0.0, 0.0], dtype=anp.float64)
        np.testing.assert_allclose(result, expected)

    def test_sigmoid_large_number_of_elements(self):
        """Test sigmoid function with a large number of elements."""
        x = anp.ones(10**6)
        result = sigmoid(x)
        expected = anp.ones(10**6, dtype=anp.float64) * 0.73105858
        np.testing.assert_allclose(result, expected)


def test_squeeze_all():
    """Test the `squeeze_all` utility function."""
    # happy path
    x = anp.asarray([[1, 2, 3], [4, 5, 6]])
    result = squeeze_all(x)
    np.testing.assert_array_equal(result, x)

    x = anp.asarray([[[1, 2, 3]]])
    result = squeeze_all(x)
    excepted_result = np.squeeze(x)
    np.testing.assert_array_equal(result, excepted_result)

    x = anp.asarray([[[1, 2, 3]], [[4, 5, 6]]])
    result = squeeze_all(x)
    excepted_result = np.squeeze(x)
    np.testing.assert_array_equal(result, excepted_result)

    x = anp.asarray([[[0], [1], [2]]])
    result = squeeze_all(x)
    excepted_result = np.squeeze(x)
    np.testing.assert_array_equal(result, excepted_result)

    # edge cases
    x = anp.asarray([])
    result = squeeze_all(x)
    excepted_result = np.squeeze(x)
    np.testing.assert_array_equal(result, excepted_result)
