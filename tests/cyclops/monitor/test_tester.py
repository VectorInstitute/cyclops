"""unit tests for Tester module."""

import numpy as np
import pytest
import torch

from cyclops.models import LSTMModel, RandomForestClassifier
from cyclops.monitor.tester import DCTester, TSTester


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
    #     os.remove(model_path)
    assert 0 <= p_val <= 1


# iterate and test DCTester for all available methods and models
methods = DCTester("classifier").get_available_test_methods()


@pytest.mark.parametrize("method", methods)
def test_dctester(source_target, method):
    """Test DCTester."""
    X_source, X_target = source_target
    if method == "classifier":
        model = RandomForestClassifier()
        tester = DCTester(method, model=model)
    elif method == "spot_the_diff":
        tester = DCTester(method)
    tester.fit(X_source)
    p_val = tester.test_shift(X_target)[0]
    assert 0 <= p_val <= 1
