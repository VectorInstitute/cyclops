"""Tests for the (binary) calibration error metric."""

import array_api_compat.torch
import numpy.array_api as anp
import pytest
import torch

from cyclops.evaluate.metrics.experimental import BinaryCalibrationError
from cyclops.evaluate.metrics.experimental.functional import binary_calibration_error


@pytest.mark.parametrize("xp", [anp, array_api_compat.torch])
def test_binary_calibration_error_two_bins(xp):
    """Test binary calibration error against a hand-computed example.

    target = [0, 1, 1, 0], preds = [0.1, 0.9, 0.8, 0.3], n_bins=2.
    Bin [0, 0.5): preds=[0.1, 0.3], target=[0, 0] -> conf=0.2, acc=0.0, gap=0.2
    Bin [0.5, 1]: preds=[0.9, 0.8], target=[1, 1] -> conf=0.85, acc=1.0, gap=0.15
    ECE = 0.5 * 0.2 + 0.5 * 0.15 = 0.175
    """
    target = xp.asarray([0, 1, 1, 0])
    preds = xp.asarray([0.1, 0.9, 0.8, 0.3])
    result = binary_calibration_error(target, preds, n_bins=2)
    assert float(result) == pytest.approx(0.175, abs=1e-4)


def test_binary_calibration_error_perfect_calibration():
    """A perfectly calibrated model must have (near) zero ECE."""
    target = anp.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    preds = anp.asarray([0.0] * 5 + [1.0] * 5)
    assert float(binary_calibration_error(target, preds, n_bins=2)) == pytest.approx(
        0.0,
    )


def test_binary_calibration_error_max_norm():
    """The 'max' norm must return the largest per-bin gap (MCE)."""
    target = anp.asarray([0, 1, 1, 0])
    preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
    result = binary_calibration_error(target, preds, n_bins=2, norm="max")
    assert float(result) == pytest.approx(0.2, abs=1e-4)


def test_binary_calibration_error_norms_ordering():
    """For a fixed input, max-norm gap must be >= l2 gap >= l1 (ECE) gap."""
    target = anp.asarray([0, 1, 1, 0, 1, 0, 0, 1])
    preds = anp.asarray([0.1, 0.9, 0.8, 0.3, 0.4, 0.2, 0.6, 0.7])
    ece = float(binary_calibration_error(target, preds, n_bins=4, norm="l1"))
    l2 = float(binary_calibration_error(target, preds, n_bins=4, norm="l2"))
    mce = float(binary_calibration_error(target, preds, n_bins=4, norm="max"))
    assert ece <= l2 <= mce


def test_binary_calibration_error_ignore_index():
    """Values matching ignore_index must be excluded from binning."""
    target = anp.asarray([0, 1, 1, -1])
    preds = anp.asarray([0.1, 0.9, 0.8, 0.99])
    without_ignored = float(
        binary_calibration_error(
            anp.asarray([0, 1, 1]),
            anp.asarray([0.1, 0.9, 0.8]),
            n_bins=2,
        ),
    )
    with_ignored = float(
        binary_calibration_error(target, preds, n_bins=2, ignore_index=-1),
    )
    assert with_ignored == pytest.approx(without_ignored, abs=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_bins": 0}, "n_bins"),
        ({"n_bins": -1}, "n_bins"),
        ({"norm": "l3"}, "norm"),
    ],
)
def test_binary_calibration_error_invalid_args(kwargs, match):
    """Invalid n_bins or norm arguments must raise ValueError."""
    target = anp.asarray([0, 1])
    preds = anp.asarray([0.1, 0.9])
    with pytest.raises(ValueError, match=match):
        binary_calibration_error(target, preds, **kwargs)


def test_binary_calibration_error_invalid_target_raises():
    """Non-binary target values must raise."""
    target = anp.asarray([0, 1, 2])
    preds = anp.asarray([0.1, 0.9, 0.8])
    with pytest.raises(RuntimeError):
        binary_calibration_error(target, preds)


class TestBinaryCalibrationErrorClass:
    """Tests for the BinaryCalibrationError metric class."""

    def test_single_call(self):
        """Test single-call usage matches the functional API."""
        target = anp.asarray([0, 1, 1, 0])
        preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
        metric = BinaryCalibrationError(n_bins=2)
        assert float(metric(target, preds)) == pytest.approx(
            float(binary_calibration_error(target, preds, n_bins=2)),
        )

    def test_streaming_matches_batch(self):
        """Accumulating bin counts over multiple updates must match a batch call."""
        target = [0, 1, 1, 0, 1, 0]
        preds = [0.1, 0.9, 0.8, 0.3, 0.4, 0.2]

        batch_result = float(
            binary_calibration_error(
                anp.asarray(target),
                anp.asarray(preds),
                n_bins=4,
            ),
        )

        metric = BinaryCalibrationError(n_bins=4)
        for t, p in zip([target[:3], target[3:]], [preds[:3], preds[3:]]):
            metric.update(anp.asarray(t), anp.asarray(p))
        streaming_result = float(metric.compute())

        assert streaming_result == pytest.approx(batch_result, abs=1e-5)

    def test_torch_backend(self):
        """Test the metric works with a torch backend."""
        target = torch.tensor([0, 1, 1, 0])
        preds = torch.tensor([0.1, 0.9, 0.8, 0.3])
        metric = BinaryCalibrationError(n_bins=2)
        result = metric(target, preds)
        assert isinstance(result, torch.Tensor)
        assert float(result) == pytest.approx(0.175, abs=1e-4)
