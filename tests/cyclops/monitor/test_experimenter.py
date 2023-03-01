"""integration tests for Experimenter module."""

import pytest
from synthetic_datasets import synthetic_gemini_dataset, synthetic_nih_dataset

from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator
from cyclops.monitor.detector import Detector
from cyclops.monitor.experimenter import Experimenter
from cyclops.monitor.reductor import Reductor
from cyclops.monitor.synthetic_applicator import SyntheticShiftApplicator
from cyclops.monitor.tester import TSTester


@pytest.fixture(name="gemini_dataset")
def fixture_gemini_dataset():
    """Create a test input for GEMINI use-case."""
    dataset = synthetic_gemini_dataset()
    return dataset


@pytest.fixture(name="nih_dataset")
def fixture_nih_dataset():
    """Create a test input for NIH use-case."""
    dataset = synthetic_nih_dataset()
    return dataset


# test gemini use-case with pca reductor and mmd tester and
# clinical shift applicator for hospital_type w/ synthetic_gemini_dataset


@pytest.mark.skip(reason="will deprecate experimenter")
@pytest.mark.integration_test
def test_experimenter_gemini_pca_mmd(gemini_dataset):
    """Test Experimenter."""
    dataset = gemini_dataset
    reductor = Reductor("PCA", n_components=2)
    tester = TSTester("mmd")
    detector = Detector(reductor, tester)
    source = ["2015-01-01", "2017-06-01"]
    target = ["2017-06-01", "2020-01-01"]
    applicator = ClinicalShiftApplicator("time", source, target)
    experimenter = Experimenter("sensitivity_test", detector, applicator)
    results = experimenter.run(dataset)

    assert list(results.keys()) == [
        "samples",
        "mean_p_vals",
        "std_p_vals",
        "mean_dist",
        "std_dist",
    ]


# test nih use-case with tae_trxv_cnn reductor and mmd tester and
# synthetic shift applicator for gaussian noise w/ txrv_dataset
@pytest.mark.skip(reason="will deprecate experimenter")
@pytest.mark.integration_test
def test_experimenter_nih_tae_trxv_cnn_mmd(nih_dataset):
    """Test Experimenter."""
    reductor = Reductor("TAE_txrv_CNN")
    tester = TSTester("mmd")
    dataset = nih_dataset
    detector = Detector(reductor, tester)
    applicator = SyntheticShiftApplicator("gn_shift")
    experimenter = Experimenter("sensitivity_test", detector, applicator)
    results = experimenter.run(dataset)

    assert list(results.keys()) == [
        "samples",
        "mean_p_vals",
        "std_p_vals",
        "mean_dist",
        "std_dist",
    ]
