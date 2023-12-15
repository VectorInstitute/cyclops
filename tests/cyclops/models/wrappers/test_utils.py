"""Tests for utility functions in wrappers subpackage in models package."""

import numpy as np
import pytest
import torch
from datasets import Dataset

from cyclops.models.wrappers.utils import (
    DatasetColumn,
    get_params,
    set_params,
    to_numpy,
    to_tensor,
)


def test_set_params():
    """Test set_params function."""

    class ExampleClass:
        """Example class for testing."""

        def __init__(self, param1, param2, param3):
            """Initialize the class."""
            self.param1 = param1
            self.param2 = param2
            self.param3 = param3

    params = {"param1": 10, "param2": "hello", "param3": True}
    example_class = ExampleClass(1, "world", False)
    set_params(example_class, **params)
    assert example_class.param1 == 10
    assert example_class.param2 == "hello"
    assert example_class.param3 is True


def test_get_params():
    """Test get_params function."""

    class ExampleClass:
        """Example class for testing."""

        def __init__(self, param1, param2, param3):
            """Initialize the class."""
            self.param1 = param1
            self.param2 = param2
            self.param3 = param3

    result = get_params(ExampleClass(10, "hello", True))
    assert isinstance(result, dict)
    assert len(result) == 3
    assert result["param1"] == 10
    assert result["param2"] == "hello"
    assert result["param3"] is True


@pytest.mark.integration_test()
def test_to_tensor():
    """Test to_tensor function."""
    test_array = np.random.rand(10, 10)
    tensor = to_tensor(test_array)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (10, 10)
    test_tensor = torch.rand(10, 10)
    tensor = to_tensor(test_tensor)
    assert isinstance(tensor, torch.Tensor)
    tensor = to_tensor(tensor, device="cuda")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device == torch.device("cuda", 0)
    test_list = [1, 2, 3]
    tensor_list = to_tensor(test_list)
    assert isinstance(tensor_list, list)
    for tensor in tensor_list:
        assert isinstance(tensor, torch.Tensor)
    test_dict = {"a": 1, "b": 2}
    tensor_dict = to_tensor(test_dict)
    assert isinstance(tensor_dict, dict)
    for _, tensor in tensor_dict.items():
        assert isinstance(tensor, torch.Tensor)
    with pytest.raises(ValueError):
        to_tensor(None)


def test_dataset_column():
    """Test DatasetColumn class."""
    test_dataset = Dataset.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
    column = DatasetColumn(test_dataset, "a")
    assert isinstance(column, DatasetColumn)
    assert len(column) == 3
    assert column[0] == 1
    for item in column:
        assert isinstance(item, int)


@pytest.mark.integration_test()
def test_to_numpy():
    """Test to_numpy function."""
    test_array = np.random.rand(10, 10)
    numpy_array = to_numpy(test_array)
    assert isinstance(numpy_array, np.ndarray)
    assert numpy_array.shape == (10, 10)
    test_tensor = torch.rand(10, 10)
    test_tensor.requires_grad = True
    numpy_array = to_numpy(test_tensor)
    assert isinstance(numpy_array, np.ndarray)
    test_cuda_tensor = torch.rand(10, 10).cuda()
    numpy_array = to_numpy(test_cuda_tensor)
    assert isinstance(numpy_array, np.ndarray)
    test_list = [1, 2, 3]
    numpy_list = to_numpy(test_list)
    assert isinstance(numpy_list, list)
    for array in numpy_list:
        assert isinstance(array, np.ndarray)
    test_dict = {"a": 1, "b": 2}
    numpy_dict = to_numpy(test_dict)
    assert isinstance(numpy_dict, dict)
    for _, array in numpy_dict.items():
        assert isinstance(array, np.ndarray)
    with pytest.raises(ValueError):
        to_numpy(None)
