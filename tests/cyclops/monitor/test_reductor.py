"""Unit tests for Reductor module."""
import pytest
from synthetic_datasets import (
    synthetic_gemini_dataset,
    synthetic_generic_dataset,
    synthetic_nih_dataset,
)
from torchxrayvision.models import DenseNet

from cyclops.models import LSTMModel
from cyclops.monitor import Reductor


@pytest.fixture(name="generic_dataset")
def fixture_generic_dataset():
    """Create a test input for NIH use-case."""
    dataset = synthetic_generic_dataset()
    return dataset


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


def test_reductor_nored(generic_dataset):
    """Test Reductor."""
    reductor = Reductor("nored")
    reductor.fit(generic_dataset)
    X_reduced = reductor.transform(generic_dataset)
    assert X_reduced.shape == (100, 10)


def test_reductor_pca(generic_dataset):
    """Test Reductor."""
    reductor = Reductor("pca", n_components=2)
    reductor.fit(generic_dataset)
    X_reduced = reductor.transform(generic_dataset)
    assert X_reduced.shape == (100, 2)


def test_reductor_srp(generic_dataset):
    """Test Reductor."""
    reductor = Reductor("srp", n_components=2)
    reductor.fit(generic_dataset)
    X_reduced = reductor.transform(generic_dataset)
    assert X_reduced.shape == (100, 2)


def test_reductor_kpca(generic_dataset):
    """Test Reductor."""
    reductor = Reductor("kpca", n_components=2)
    reductor.fit(generic_dataset)
    X_reduced = reductor.transform(generic_dataset)
    assert X_reduced.shape == (100, 2)


def test_reductor_isomap(generic_dataset):
    """Test Reductor."""
    reductor = Reductor("isomap", n_components=2)
    reductor.fit(generic_dataset)
    X_reduced = reductor.transform(generic_dataset)
    assert X_reduced.shape == (100, 2)


def test_reductor_gmm(generic_dataset):
    """Test Reductor."""
    reductor = Reductor("gmm", n_components=2)
    reductor.fit(generic_dataset)
    X_reduced = reductor.transform(generic_dataset)
    assert X_reduced.shape == (100, 2)


def test_reductor_bbsds_untrained_lstm(gemini_dataset):
    """Test Reductor."""
    model = LSTMModel(7)
    reductor = Reductor("bbse-soft", model=model)
    reductor.fit(gemini_dataset)
    X_reduced = reductor.transform(gemini_dataset)
    assert X_reduced.shape == (100, 64, 1)


def test_reductor_bbsd_txrv_cnn(nih_dataset):
    """Test Reductor."""
    model = DenseNet(weights="densenet121-res224-all")
    reductor = Reductor("bbse-soft", model=model)
    reductor.fit(nih_dataset)
    X_reduced = reductor.transform(nih_dataset)
    assert X_reduced.shape == (8, 18)
