"""Test import of subpackages with base cyclops install."""

import numpy as np
import pytest

from cyclops.models import create_model


def test_import_cyclops():
    """Test import of cyclops."""
    import cyclops

    assert cyclops.__name__ == "cyclops"

    import cyclops.data

    assert cyclops.data.__name__ == "cyclops.data"

    import cyclops.process

    assert cyclops.process.__name__ == "cyclops.process"

    import cyclops.evaluate

    assert cyclops.evaluate.__name__ == "cyclops.evaluate"

    import cyclops.models

    assert cyclops.models.__name__ == "cyclops.models"

    import cyclops.monitor

    assert cyclops.monitor.__name__ == "cyclops.monitor"


def test_medical_image_feature_without_monai():
    """Test that the MedicalImage feature raises an error without MONAI installed."""
    from cyclops.data.features.medical_image import MedicalImage

    feat = MedicalImage()
    # create a dummy image
    img = np.random.rand(10, 10, 10)

    # test encode_example
    with pytest.raises(
        RuntimeError,
        match="The MONAI library is required to use the `MedicalImage` feature.*",
    ):
        feat.encode_example(img)

    # test decode_example
    dummy_val = {"path": "/dummy/local/path", "bytes": None}
    with pytest.raises(
        RuntimeError,
        match="The MONAI library is required to use the `MedicalImage` feature.*",
    ):
        feat.decode_example(dummy_val)


def test_model_catalog_without_xgboost():
    """Test that the ModelCatalog raises an error without XGBoost installed."""
    with pytest.raises(
        RuntimeError,
        match="The XGBoost library is required to use the `XGBClassifier` model.*",
    ):
        create_model("xgb_classifier")


def test_model_catalog_without_torchxrayvision():
    """Test that the ModelCatalog raises an error without torchxrayvision installed."""
    with pytest.raises(
        RuntimeError,
        match="The torchxrayvision library is required to use the `DenseNet` or `ResNet` model.*",
    ):
        create_model("densenet")
        create_model("resnet")


def test_model_catalog_without_pytorch():
    """Test that the ModelCatalog raises an error without PyTorch installed."""
    with pytest.raises(
        RuntimeError,
        match="The PyTorch library is required to use the `DenseNet` or `ResNet` model.*",
    ):
        create_model("gru")
        create_model("lstm")
