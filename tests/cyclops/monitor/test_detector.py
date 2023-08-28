"""Integration tests for Reductor module."""

import pytest
from synthetic_datasets import (
    synthetic_gemini_dataset,
    synthetic_generic_dataset,
    synthetic_nih_dataset,
)

from cyclops.monitor.detector import Detector
from cyclops.monitor.reductor import Reductor
from cyclops.monitor.tester import TSTester


@pytest.fixture(name="gemini_dataset")
def fixture_gemini_dataset():
    """Create a test input for GEMINI use-case."""
    return synthetic_gemini_dataset()


@pytest.fixture(name="nih_dataset")
def fixture_nih_dataset():
    """Create a test input for NIH use-case."""
    return synthetic_nih_dataset()


@pytest.fixture(name="source_target")
def fixture_source_target():
    """Create a test input."""
    ds_source = synthetic_generic_dataset()
    ds_target = synthetic_generic_dataset()
    return ds_source, ds_target


def test_detector_pca_mmd(source_target):
    """Test Detector."""
    reductor = Reductor("pca", n_components=2)
    tester = TSTester("mmd")
    detector = Detector(
        "sensitivity_test",
        reductor,
        tester,
        source_sample_size=10,
        target_sample_size=[2, 5, 10],
        num_runs=2,
    )
    ds_source, ds_target = source_target
    results = detector.detect_shift(ds_source, ds_target)
    assert results["p_val"].shape == (2, 3)
