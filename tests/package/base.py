"""Test import of subpackages with base cyclops install."""

import numpy as np
import pytest


def test_import_cyclops():
    """Test import of cyclops."""
    import cyclops

    assert cyclops.__name__ == "cyclops"

    from cyclops import data

    assert data.__name__ == "cyclops.data"

    with pytest.raises(ImportError):
        import cyclops.evaluate
    with pytest.raises(ImportError):
        import cyclops.models
    with pytest.raises(ImportError):
        import cyclops.monitor


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
