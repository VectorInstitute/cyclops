"""unit tests for Synthetic Applicator."""

import pytest

from cyclops.monitor.datasets.utils import synthetic_gemini_dataset
from cyclops.monitor.synthetic_applicator import SyntheticShiftApplicator


@pytest.fixture(name="dataset")
def fixture_dataset():
    """Create a test input."""
    metadata, features, metadata_mapping = synthetic_gemini_dataset()
    return features.reshape(features.shape[0], -1), metadata, metadata_mapping


# Test all shift types


def test_sa_gn_shift(dataset):
    """Test SyntheticShiftApplicator gaussian shift."""
    features, metadata, metadata_mapping = dataset
    applicator = SyntheticShiftApplicator("gn_shift")
    X_s, _ = applicator.apply_shift(features, metadata, metadata_mapping)
    assert X_s.shape[-1] == features.shape[-1]
