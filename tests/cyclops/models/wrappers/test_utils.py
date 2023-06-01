"""Tests for utility functions in wrappers subpackage in models package."""

import numpy as np
import pytest
import torch
from datasets import Dataset

from cyclops.models.wrappers.utils import DatasetColumn, to_numpy, to_tensor


@pytest.mark.integration_test
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
