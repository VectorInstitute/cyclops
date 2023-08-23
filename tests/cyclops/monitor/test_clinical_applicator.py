"""unit tests for Clinical Applicator."""

import pytest
from synthetic_datasets import synthetic_gemini_dataset

from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator


@pytest.fixture(name="gemini_dataset")
def fixture_gemini_dataset():
    """Create a test input for GEMINI use-case."""
    return synthetic_gemini_dataset()


def test_csa_time(gemini_dataset):
    """Test ClinicalShiftApplicator time shift."""
    source = ["2015-01-01", "2017-06-01"]
    target = ["2017-06-01", "2020-01-01"]
    applicator = ClinicalShiftApplicator(
        "time", source, target, shift_id="admit_timestamp",
    )
    X_s, X_t = applicator.apply_shift(gemini_dataset)
    assert X_t.shape[-1] == X_s.shape[-1]


def test_csa_month(gemini_dataset):
    """Test ClinicalShiftApplicator month shift."""
    source = [1, 3, 5, 7, 9, 11]
    target = [2, 4, 6, 8, 10, 12]
    applicator = ClinicalShiftApplicator(
        "month", source, target, shift_id="admit_timestamp",
    )
    X_s, X_t = applicator.apply_shift(gemini_dataset)
    assert X_t.shape[-1] == X_s.shape[-1]


def test_csa_hospital_type(gemini_dataset):
    """Test ClinicalShiftApplicator hospital_type shift."""
    source = ["MSH", "PMH", "SMH", "UHNTW", "UHNTG", "SBK"]
    target = ["THPC", "THPM"]
    applicator = ClinicalShiftApplicator(
        "hospital_type", source, target, shift_id="hospital_id",
    )
    X_s, X_t = applicator.apply_shift(gemini_dataset)
    assert X_t.shape[-1] == X_s.shape[-1]
