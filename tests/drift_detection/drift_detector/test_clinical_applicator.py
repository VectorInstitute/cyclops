"""unit tests for Clinical Applicator."""

import pytest

from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator
from cyclops.monitor.datasets.utils import synthetic_gemini_dataset


@pytest.fixture(name="dataset")
def fixture_dataset():
    """Create a test input."""
    metadata, features, metadata_mapping = synthetic_gemini_dataset()
    return features.reshape(features.shape[0], -1), metadata, metadata_mapping


def test_csa_time(dataset):
    """Test ClinicalShiftApplicator time shift."""
    features, metadata, metadata_mapping = dataset
    source = ["2015-01-01", "2017-06-01"]
    target = ["2017-06-01", "2020-01-01"]
    applicator = ClinicalShiftApplicator("time", source, target)
    X_s, X_t = applicator.apply_shift(features, metadata, metadata_mapping)
    assert X_t.shape[-1] == X_s.shape[-1]


def test_csa_month(dataset):
    """Test ClinicalShiftApplicator month shift."""
    features, metadata, metadata_mapping = dataset
    source = [1, 3, 5, 7, 9, 11]
    target = [2, 4, 6, 8, 10, 12]
    applicator = ClinicalShiftApplicator("month", source, target)
    X_s, X_t = applicator.apply_shift(
        features,
        metadata,
        metadata_mapping,
    )
    assert X_t.shape[-1] == X_s.shape[-1]


def test_csa_hospital_type(dataset):
    """Test ClinicalShiftApplicator hospital_type shift."""
    features, metadata, metadata_mapping = dataset
    source = metadata_mapping["hospital_type_academic"]
    target = metadata_mapping["hospital_type_community"]
    applicator = ClinicalShiftApplicator("hospital_type", source, target)
    X_s, X_t = applicator.apply_shift(features, metadata, metadata_mapping)
    assert X_t.shape[-1] == X_s.shape[-1]
