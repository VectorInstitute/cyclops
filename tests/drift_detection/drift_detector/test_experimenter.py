"""integration tests for Experimenter module."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from drift_detection.datasets.utils import synthetic_gemini_dataset
from drift_detection.drift_detector.clinical_applicator import ClinicalShiftApplicator
from drift_detection.drift_detector.detector import Detector
from drift_detection.drift_detector.experimenter import Experimenter
from drift_detection.drift_detector.reductor import Reductor
from drift_detection.drift_detector.synthetic_applicator import SyntheticShiftApplicator
from drift_detection.drift_detector.tester import TSTester


@pytest.fixture(name="gemini_dataset")
def fixture_gemini_dataset():
    """Create a test input for GEMINI use-case."""
    metadata, features, metadata_mapping = synthetic_gemini_dataset()
    return features.reshape(features.shape[0], -1), metadata, metadata_mapping


@pytest.fixture(name="txrv_dataset")
def fixture_txrv_dataset():
    """Create a test input for NIH use-case."""

    class TXRVDataset(Dataset):
        """TXRV Dummy Dataset."""

        def __init__(self, num_samples, channels, height, width, num_labels=14):
            self.len = num_samples
            self.data = torch.rand(num_samples, channels, height, width)
            self.labels = torch.rand(num_samples, num_labels)

        def __getitem__(self, index):
            item = {"img": self.data[index], "lab": self.labels[index]}
            return item

        def __len__(self):
            return self.len

    dataset = TXRVDataset(100, 1, 224, 224)
    metadata = pd.DataFrame(np.random.randint(0, 2, size=(100, 2)), columns=list("AB"))
    metadata_mapping = {"A": "A", "B": "B"}
    return dataset, metadata, metadata_mapping


# test gemini use-case with pca reductor and mmd tester and
# clinical shift applicator for hospital_type w/ synthetic_gemini_dataset
@pytest.mark.integration_test
def test_experimenter_gemini_pca_mmd(gemini_dataset):
    """Test Experimenter."""
    X, metadata, metadata_mapping = gemini_dataset
    reductor = Reductor("PCA", n_components=2)
    tester = TSTester("mmd")
    detector = Detector(reductor, tester)
    source = ["2015-01-01", "2017-06-01"]
    target = ["2017-06-01", "2020-01-01"]
    applicator = ClinicalShiftApplicator("time", source, target)
    experimenter = Experimenter("sensitivity_test", detector, applicator)
    results = experimenter.run(X, metadata, metadata_mapping)

    assert list(results.keys()) == [
        "samples",
        "mean_p_vals",
        "std_p_vals",
        "mean_dist",
        "std_dist",
    ]


# test nih use-case with tae_trxv_cnn reductor and mmd tester and
# synthetic shift applicator for gaussian noise w/ txrv_dataset
@pytest.mark.integration_test
def test_experimenter_nih_tae_trxv_cnn_mmd(txrv_dataset):
    """Test Experimenter."""
    reductor = Reductor("TAE_txrv_CNN")
    tester = TSTester("mmd")
    dataset, metadata, metadata_mapping = txrv_dataset
    detector = Detector(reductor, tester)
    applicator = SyntheticShiftApplicator("gn_shift")
    experimenter = Experimenter("sensitivity_test", detector, applicator)
    results = experimenter.run(dataset, metadata, metadata_mapping)

    assert list(results.keys()) == [
        "samples",
        "mean_p_vals",
        "std_p_vals",
        "mean_dist",
        "std_dist",
    ]
