"""Unit tests for Reductor module."""
import numpy as np
import pytest
from synthetic_datasets import synthetic_nih_dataset

from cyclops.monitor import Reductor


@pytest.fixture(name="X")
def fixture_x():
    """Create a test input."""
    x = np.random.rand(100, 10)
    return x


@pytest.fixture(name="X_timeseries")
def fixture_x_timeseries():
    """Create a test input."""
    x = np.random.rand(100, 32, 10)
    return x


@pytest.fixture(name="nih_dataset")
def fixture_nih_dataset():
    """Create a test input for NIH use-case."""
    dataset = synthetic_nih_dataset()
    return dataset


@pytest.fixture(name="image_dataset")
def fixture_image_dataset():
    """Create a test input."""
    x = np.random.rand(100, 3, 32, 32)
    return x


def test_reductor_nored(X):
    """Test Reductor."""
    reductor = Reductor("NoRed")
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 10)


def test_reductor_pca(X):
    """Test Reductor."""
    reductor = Reductor("PCA", n_components=2)
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 2)


def test_reductor_srp(X):
    """Test Reductor."""
    reductor = Reductor("SRP", n_components=2)
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 2)


def test_reductor_kpca(X):
    """Test Reductor."""
    reductor = Reductor("kPCA", n_components=2)
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 2)


def test_reductor_isomap(X):
    """Test Reductor."""
    reductor = Reductor("Isomap", n_components=2)
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 2)


def test_reductor_gmm(X):
    """Test Reductor."""
    reductor = Reductor("GMM", n_components=2)
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 2)


def test_reductor_bbsds_untrained_ffnn(X):
    """Test Reductor."""
    reductor = Reductor("BBSDs_untrained_FFNN", n_features=10, n_classes=2)
    reductor.fit(X)
    X_reduced, _ = reductor.transform(X)
    assert X_reduced.shape == (100, 2)


def test_reductor_bbsds_untrained_cnn(image_dataset):
    """Test Reductor."""
    reductor = Reductor("BBSDs_untrained_CNN", n_features=3, n_classes=2)
    reductor.fit(image_dataset)
    X_reduced, _ = reductor.transform(image_dataset)
    assert X_reduced.shape == (100, 2)


def test_reductor_bbsds_untrained_lstm(X_timeseries):
    """Test Reductor."""
    reductor = Reductor("BBSDs_untrained_LSTM", n_features=10)
    reductor.fit(X_timeseries)
    X_reduced, _ = reductor.transform(X_timeseries)
    assert X_reduced.shape == (100, 32, 1)


def test_reductor_bbsd_txrv_cnn(nih_dataset):
    """Test Reductor."""
    reductor = Reductor("BBSDs_txrv_CNN")
    reductor.fit(nih_dataset)
    X_reduced, _ = reductor.transform(nih_dataset)
    assert X_reduced.shape == (8, 18)
