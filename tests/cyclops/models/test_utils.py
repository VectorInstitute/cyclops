"""Test utils module in models package."""

import pytest

from cyclops.models.utils import LossMeter, get_module


def test_get_module():
    """Test get_module function."""
    module = get_module("criterion", "BCELoss")
    assert module.__name__ == "BCELoss"
    assert module.__module__ == "torch.nn.modules.loss"
    module = get_module("optimizer", "Adam")
    assert module.__name__ == "Adam"
    assert module.__module__ == "torch.optim.adam"
    module = get_module("lr_scheduler", "StepLR")
    assert module.__name__ == "StepLR"
    assert module.__module__ == "torch.optim.lr_scheduler"
    module = get_module("activation", "ReLU")
    assert module.__name__ == "ReLU"
    assert module.__module__ == "torch.nn.modules.activation"
    with pytest.raises(ValueError):
        _ = get_module("donkey", "monkey")
    with pytest.raises(ValueError):
        _ = get_module("optimizer", "monkey")


def test_loss_meter():
    """Test LossMeter class."""
    loss_meter = LossMeter("train")
    loss_meter.add(1.0)
    loss_meter.add(2.0)
    loss_meter.add(3.0)
    assert loss_meter.mean() == 2.0
    loss_meter.reset()
    assert loss_meter.mean() == 0.0
    loss_meter.add(1.0)
    loss_meter.add(2.0)
    assert loss_meter.sum() == 3.0
    assert loss_meter.pop() == 2.0
