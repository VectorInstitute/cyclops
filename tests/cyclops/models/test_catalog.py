"""Test model catalog module."""

import pytest

from cyclops.models.catalog import create_model, list_models


def test_list_models():
    """Test list_models function."""
    models = list_models("static")
    assert "sgd_classifier" in models
    models = list_models("temporal")
    assert "lstm" in models
    models = list_models("pytorch")
    assert "resnet" in models
    models = list_models("sklearn")
    assert "sgd_classifier" in models
    models = list_models("image")
    assert "resnet" in models
    with pytest.raises(ValueError):
        list_models("foo")


def test_wrap_model():
    """Test wrap_model function."""
    model = create_model("sgd_classifier")
    assert model.model.__name__ == "SGDClassifier"
    model = create_model("lstm")
    assert model.model.__name__ == "LSTMModel"
