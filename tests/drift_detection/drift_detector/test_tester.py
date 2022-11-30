"""unit tests for Tester module."""

import os

import numpy as np
import pytest
import torch

from drift_detection.drift_detector.tester import DCTester, TSTester
from drift_detection.drift_detector.utils import recurrent_neural_network


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
        X_source, X_target = np.random.rand(100, 32, 10), np.random.rand(100, 32, 10)
        model = recurrent_neural_network("lstm", X_source.shape[-1])
        model_path = "./model.pt"
        torch.save({"model": model.state_dict()}, model_path)
        tester = TSTester(method, model_path=model_path)
    else:
        tester = TSTester(method)
    if method == "fet":
        # source must be binary
        X_source = np.random.randint(0, 2, size=(100, 10))
    tester.fit(X_source)
    p_val = tester.test_shift(X_target)[0]
    if isinstance(p_val, np.ndarray):
        p_val = p_val.min()
    if method == "ctx_mmd":
        os.remove(model_path)
    assert 0 <= p_val <= 1


# iterate and test DCTester for all available methods and models
methods = DCTester("classifier", "rf").get_available_test_methods()
models = DCTester("classifier", "rf").get_available_model_methods()


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("method", methods)
def test_dctester(source_target, method, model):
    """Test DCTester."""
    tester = DCTester(method, model)
    X_source, X_target = source_target
    if model == "rnn" and method == "classifier":
        # fix this test
        pytest.skip(
            "lstm not supported for classifier. \
            Must squeeze last dim of binary classification output."
        )
    elif model == "cnn":
        X_source, X_target = np.random.rand(100, 3, 32, 32), np.random.rand(
            100, 3, 32, 32
        )
        tester.fit(X_source, num_channels=3, num_classes=10)
    elif model == "ffnn":
        tester.fit(X_source, num_features=X_source.shape[-1], num_classes=10)
    else:
        tester.fit(X_source)

    p_val = tester.test_shift(X_target)[0]
    assert 0 <= p_val <= 1
