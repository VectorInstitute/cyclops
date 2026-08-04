"""Integration tests for Reductor module."""

import pytest
from synthetic_datasets import (
    synthetic_gemini_dataset,
    synthetic_generic_dataset,
    synthetic_nih_dataset,
)

from cyclops.data.slicer import SliceSpec
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
    reductor = Reductor(
        "pca",
        n_components=2,
        feature_columns=[f"feature_{i}" for i in range(10)],
    )
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


def test_detector_detect_shift_by_subgroup(source_target):
    """Test Detector.detect_shift_by_subgroup."""
    reductor = Reductor(
        "pca",
        n_components=2,
        feature_columns=[f"feature_{i}" for i in range(10)],
    )
    tester = TSTester("mmd")
    detector = Detector("sensitivity_test", reductor, tester)
    ds_source, ds_target = source_target
    detector.fit(ds_source)

    slice_spec = SliceSpec(
        spec_list=[{"mortality": {"value": 0}}, {"mortality": {"value": 1}}],
        include_overall=False,
    )
    results = detector.detect_shift_by_subgroup(ds_target, slice_spec)

    assert set(results.keys()) == set(slice_spec.get_slices().keys())
    for subgroup_result in results.values():
        assert subgroup_result["sample_size"] > 0
        assert 0 <= subgroup_result["p_val"] <= 1
        assert subgroup_result["shift_detected"] in (0, 1)


def test_detector_detect_shift_by_subgroup_small_subgroup_skipped(source_target):
    """Subgroups below min_sample_size must be skipped, not tested."""
    reductor = Reductor(
        "pca",
        n_components=2,
        feature_columns=[f"feature_{i}" for i in range(10)],
    )
    tester = TSTester("mmd")
    detector = Detector("sensitivity_test", reductor, tester)
    ds_source, ds_target = source_target
    detector.fit(ds_source)

    slice_spec = SliceSpec(
        spec_list=[{"mortality": {"value": 0}}],
        include_overall=False,
    )
    results = detector.detect_shift_by_subgroup(
        ds_target,
        slice_spec,
        min_sample_size=10_000,
    )

    (subgroup_result,) = results.values()
    assert subgroup_result["p_val"] is None
    assert subgroup_result["shift_detected"] is None
    assert subgroup_result["sample_size"] < 10_000


def test_detector_detect_shift_by_subgroup_invalid_correction(source_target):
    """An unknown correction method must raise a clear error."""
    reductor = Reductor(
        "pca",
        n_components=2,
        feature_columns=[f"feature_{i}" for i in range(10)],
    )
    tester = TSTester("mmd")
    detector = Detector("sensitivity_test", reductor, tester)
    ds_source, ds_target = source_target
    detector.fit(ds_source)

    slice_spec = SliceSpec(spec_list=[{"mortality": {"value": 0}}])
    with pytest.raises(ValueError, match="correction"):
        detector.detect_shift_by_subgroup(ds_target, slice_spec, correction="invalid")
