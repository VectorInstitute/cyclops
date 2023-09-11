"""Unit tests for Tester module."""

import numpy as np
import pytest
import torch
from synthetic_datasets import synthetic_generic_dataset
from torch import nn

from cyclops.models import LSTMModel, RandomForestClassifier
from cyclops.monitor.tester import DCTester, TSTester


@pytest.fixture(name="generic_source_target")
def fixture_generic_source_target():
    """Create a test input."""
    X_source = synthetic_generic_dataset(concatenate_features=True)
    X_target = synthetic_generic_dataset(concatenate_features=True)
    return X_source, X_target


@pytest.fixture(name="source_target")
def fixture_source_target():
    """Create a test input."""
    X_source = np.random.rand(100, 10)
    X_target = np.random.rand(100, 10)
    return X_source, X_target


# iterate and test TSTester for all available methods
methods = TSTester("mmd").get_available_test_methods()


@pytest.mark.parametrize("method", methods)
def test_tstester(source_target, method):
    """Test TSTester."""
    X_source, X_target = source_target
    if method == "ctx_mmd":
        pytest.skip("Have to re-factor ctx_mmd to use the new model loading system.")
        X_source, X_target = np.random.rand(100, 32, 10), np.random.rand(100, 32, 10)
        model = LSTMModel(X_source.shape[-1])
        model_path = "./model.pt"
        torch.save({"model": model.state_dict()}, model_path)
        tester = TSTester(method, model_path=model_path)
    elif method == "lk":
        model = torch.nn.Linear(10, 1)
        tester = TSTester(method, projection=model)
    else:
        tester = TSTester(method)
    if method == "fet":
        # source must be binary
        X_source = np.random.randint(0, 2, size=(100, 10))
    tester.fit(X_source)
    p_val = tester.test_shift(X_target)[0]
    if isinstance(p_val, np.ndarray):
        p_val = p_val.min()
    # if method == "ctx_mmd":
    assert 0 <= p_val <= 1


# iterate and test DCTester for all available methods and models
methods = DCTester("classifier").get_available_test_methods()


@pytest.mark.parametrize("method", methods)
def test_dctester(source_target, generic_source_target, method):
    """Test DCTester."""
    if method == "classifier":
        X_source, X_target = source_target
        model = RandomForestClassifier()
        tester = DCTester(method, model=model)
    elif method == "detectron":
        X_source, X_target = generic_source_target
        base_model = torch.nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        tester = DCTester(
            method,
            base_model=base_model,
            feature_column="features",
            task="multiclass",
            sample_size=10,
            max_epochs_per_model=1,
            ensemble_size=1,
            lr=0.01,
            num_runs=1,
        )
    else:
        X_source, X_target = source_target
        tester = DCTester(method)
    tester.fit(X_source)
    p_val = tester.test_shift(X_target)[0]
    assert 0 <= p_val <= 1
