'''integration tests for Experimenter module'''

import numpy as np
import torch
import pytest

from drift_detection.drift_detector.experimenter import Experimenter
from drift_detection.drift_detector.synthetic_applicator import SyntheticShiftApplicator
from drift_detection.drift_detector.clinical_applicator import ClinicalShiftApplicator
from drift_detection.drift_detector.reductor import Reductor
from drift_detection.drift_detector.tester import DCTester, TSTester
from drift_detection.drift_detector.detector import Detector

from drift_detection.datasets.utils import synthetic_gemini_dataset

@pytest.fixture
def synthetic_gemini_dataset():
    '''create a test input'''
    admin_data, X, metadata_mapping = synthetic_gemini_dataset()
    
    return admin_data, X, metadata_mapping

@pytest.fixture
def txrv_dataset():
    '''create a test input'''
    import torch
    from torch.utils.data import Dataset

    class TXRVDataset(Dataset):
        def __init__(self, num_samples, channels, height, width, num_labels=14):
            self.len = num_samples
            self.data = torch.rand(num_samples, channels, height, width)
            self.num_labels = torch.rand(num_samples, num_labels)

        def __getitem__(self, index):
            item = {"img": self.data[index], "lab": torch.rand(self.num_labels[index])}
            return item
        def __len__(self):
            return self.len

    dataset = TXRVDataset(100, 1, 224, 224)
    return dataset


# test gemini use-case with pca reductor and mmd tester and 
# clinical shift applicator for hospital_type w/ synthetic_gemini_dataset
@pytest.mark.integtest
def test_experimenter_gemini_pca_mmd(synthetic_gemini_dataset):
    '''test Experimenter'''
    admin_data, X, metadata_mapping = synthetic_gemini_dataset
    reductor = Reductor("PCA", n_components=2)
    tester = TSTester("mmd")
    detector = Detector(reductor, tester)
    applicator = ClinicalShiftApplicator("hospital type")
    experimenter = Experimenter('sensitivity_test', detector, applicator)
    results = experimenter.run(X, admin_data, metadata_mapping)
    
    list(results.keys()) == ['samples', 'mean_p_vals', 'std_p_vals', 'mean_dist', 'std_dist']


# test nih use-case with tae_trxv_cnn reductor and mmd tester and
# synthetic shift applicator for gaussian noise w/ txrv_dataset
@pytest.mark.integtest
def test_experimenter_nih_tae_trxv_cnn_mmd(txrv_dataset):
    '''test Experimenter'''
    reductor = Reductor("tae_trxv_cnn")
    tester = TSTester("mmd")
    detector = Detector(reductor, tester)
    applicator = SyntheticShiftApplicator("gaussian noise")
    experimenter = Experimenter('sensitivity_test', detector, applicator)
    results = experimenter.run(txrv_dataset)
    
    list(results.keys()) == ['samples', 'mean_p_vals', 'std_p_vals', 'mean_dist', 'std_dist']

