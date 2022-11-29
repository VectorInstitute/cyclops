'''unit tests for Reductor module'''
from drift_detection.drift_detector import Reductor


import numpy as np
import torch
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drift_detection.drift_detector.reductor import Reductor


@pytest.fixture
def X():
    '''create a test input'''
    X = np.random.rand(100, 10)
    return X

# # pytest fixture for torch dataset of random samples
# @pytest.fixture
# def torch_dataset():
#     '''create a test input'''
#     X = np.random.rand(100, 10)
#     return torch.utils.data.TensorDataset(torch.from_numpy(X))

# pytest fixture for torch dataset of random images and labels in dict with keys "img" and "lab"
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

'''test all reductor methods: 
"NoRed" "PCA" "SRP" "kPCA" "Isomap" "GMM" "BBSDs_untrained_FFNN" 
"BBSDh_untrained_FFNN" "BBSDs_untrained_CNN" "BBSDh_untrained_CNN" 
"BBSDs_untrained_LSTM" "BBSDh_untrained_LSTM" "BBSDs_trained_LSTM" 
"BBSDh_trained_LSTM" "BBSDs_txrv_CNN" "BBSDh_txrv_CNN" "TAE_txrv_CNN"
'''

def test_reductor_NoRed(X):
    '''test Reductor'''
    reductor = Reductor("NoRed")
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 10)

def test_reductor_pca(X):
    '''test Reductor'''
    reductor = Reductor("PCA", n_components=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_srp(X):
    '''test Reductor'''
    reductor = Reductor("SRP", n_components=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_kpca(X):
    '''test Reductor'''
    reductor = Reductor("kPCA", n_components=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_isomap(X):
    '''test Reductor'''
    reductor = Reductor("Isomap", n_components=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_gmm(X):
    '''test Reductor'''
    reductor = Reductor("GMM", gmm_n_clusters=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_bbsds_untrained_ffnn(X):
    '''test Reductor'''
    reductor = Reductor("BBSDs_untrained_FFNN", num_features=10, num_classes=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_bbsds_untrained_cnn(X):
    '''test Reductor'''
    reductor = Reductor("BBSDs_untrained_CNN", num_features=10, num_classes=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_bbsds_untrained_lstm(X):
    '''test Reductor'''
    reductor = Reductor("BBSDs_untrained_LSTM", model_name='lstm', n_features=10, n_classes=2)
    reductor.fit(X)
    X_reduced = reductor.transform(X)
    assert X_reduced.shape == (100, 2)

def test_reductor_bbsd_txrv_cnn(txrv_dataset):
    '''test Reductor'''
    reductor = Reductor("BBSDs_txrv_CNN")
    reductor.fit(txrv_dataset)
    X_reduced = reductor.transform(txrv_dataset)
    assert X_reduced.shape == (100, 2)





