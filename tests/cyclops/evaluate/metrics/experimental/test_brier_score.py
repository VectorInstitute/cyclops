"""Tests for the Brier score metric."""

import array_api_compat.torch
import numpy as np
import numpy.array_api as anp
import pytest
import torch
from sklearn.metrics import brier_score_loss

from cyclops.evaluate.metrics.experimental import (
    BinaryBrierScore,
    MulticlassBrierScore,
)
from cyclops.evaluate.metrics.experimental.functional import (
    binary_brier_score,
    multiclass_brier_score,
)


@pytest.mark.parametrize("xp", [anp, array_api_compat.torch])
def test_binary_brier_score_matches_sklearn(xp):
    """Binary Brier score must match sklearn's brier_score_loss."""
    target_list = [0, 1, 1, 0, 1, 0, 0, 1]
    preds_list = [0.1, 0.9, 0.8, 0.3, 0.4, 0.2, 0.6, 0.7]
    expected = brier_score_loss(target_list, preds_list)

    target = xp.asarray(target_list)
    preds = xp.asarray(preds_list)
    result = binary_brier_score(target, preds)

    assert float(result) == pytest.approx(expected, abs=1e-5)


def test_binary_brier_score_perfect_predictions():
    """Brier score for perfect predictions must be 0."""
    target = anp.asarray([0, 1, 0, 1])
    preds = anp.asarray([0.0, 1.0, 0.0, 1.0])
    assert float(binary_brier_score(target, preds)) == pytest.approx(0.0)


def test_binary_brier_score_worst_predictions():
    """Brier score for maximally wrong predictions must be 1."""
    target = anp.asarray([0, 1, 0, 1])
    preds = anp.asarray([1.0, 0.0, 1.0, 0.0])
    assert float(binary_brier_score(target, preds)) == pytest.approx(1.0)


def test_binary_brier_score_from_logits():
    """Logits (values outside [0, 1]) must be converted via sigmoid."""
    target = anp.asarray([0, 1, 1, 0])
    logits_np = np.asarray([-3.0, 3.0, 2.0, -1.0])
    logits = anp.asarray(logits_np)

    result = float(binary_brier_score(target, logits))
    expected = brier_score_loss([0, 1, 1, 0], 1 / (1 + np.exp(-logits_np)))
    assert result == pytest.approx(expected, abs=1e-5)


def test_binary_brier_score_ignore_index():
    """Values matching ignore_index must be excluded."""
    target = anp.asarray([0, 1, 1, -1])
    preds = anp.asarray([0.1, 0.9, 0.8, 0.99])
    result = binary_brier_score(target, preds, ignore_index=-1)
    expected = brier_score_loss([0, 1, 1], [0.1, 0.9, 0.8])
    assert float(result) == pytest.approx(expected, abs=1e-5)


def test_binary_brier_score_invalid_target_raises():
    """Non-binary target values must raise."""
    target = anp.asarray([0, 1, 2])
    preds = anp.asarray([0.1, 0.9, 0.8])
    with pytest.raises(RuntimeError):
        binary_brier_score(target, preds)


class TestBinaryBrierScoreClass:
    """Tests for the BinaryBrierScore metric class."""

    def test_single_call(self):
        """Test single-call usage matches the functional API."""
        target = anp.asarray([0, 1, 1, 0])
        preds = anp.asarray([0.1, 0.9, 0.8, 0.3])
        metric = BinaryBrierScore()
        assert float(metric(target, preds)) == pytest.approx(
            float(binary_brier_score(target, preds)),
        )

    def test_streaming_matches_batch(self):
        """Accumulating over multiple updates must match a single batch call."""
        target = [0, 1, 1, 0, 1, 0]
        preds = [0.1, 0.9, 0.8, 0.3, 0.4, 0.2]

        batch_result = float(
            binary_brier_score(anp.asarray(target), anp.asarray(preds)),
        )

        metric = BinaryBrierScore()
        for t, p in zip([target[:3], target[3:]], [preds[:3], preds[3:]]):
            metric.update(anp.asarray(t), anp.asarray(p))
        streaming_result = float(metric.compute())

        assert streaming_result == pytest.approx(batch_result, abs=1e-5)

    def test_torch_backend(self):
        """Test the metric works with a torch backend."""
        target = torch.tensor([0, 1, 1, 0])
        preds = torch.tensor([0.1, 0.9, 0.8, 0.3])
        metric = BinaryBrierScore()
        result = metric(target, preds)
        assert isinstance(result, torch.Tensor)
        assert float(result) == pytest.approx(0.0375, abs=1e-4)


def test_multiclass_brier_score_matches_manual_computation():
    """Multiclass Brier score must match a manually one-hot-encoded MSE."""
    target = anp.asarray([0, 1, 2])
    preds_np = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    preds = anp.asarray(preds_np)

    one_hot = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    expected = np.mean(np.sum((preds_np - one_hot) ** 2, axis=1))

    result = multiclass_brier_score(target, preds, num_classes=3)
    assert float(result) == pytest.approx(expected, abs=1e-5)


def test_multiclass_brier_score_perfect_predictions():
    """Multiclass Brier score for perfect one-hot predictions must be 0."""
    target = anp.asarray([0, 1, 2])
    preds = anp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert float(multiclass_brier_score(target, preds, num_classes=3)) == pytest.approx(
        0.0,
    )


def test_multiclass_brier_score_invalid_num_classes():
    """num_classes < 2 must raise."""
    target = anp.asarray([0, 1])
    preds = anp.asarray([[1.0], [1.0]])
    with pytest.raises(ValueError, match="num_classes"):
        multiclass_brier_score(target, preds, num_classes=1)


def test_multiclass_brier_score_wrong_preds_shape():
    """Preds without one more dimension than target must raise."""
    target = anp.asarray([0, 1, 2])
    preds = anp.asarray([0.1, 0.9, 0.8])
    with pytest.raises(ValueError, match="preds"):
        multiclass_brier_score(target, preds, num_classes=3)


class TestMulticlassBrierScoreClass:
    """Tests for the MulticlassBrierScore metric class."""

    def test_single_call(self):
        """Test single-call usage matches the functional API."""
        target = anp.asarray([0, 1, 2])
        preds = anp.asarray(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
        )
        metric = MulticlassBrierScore(num_classes=3)
        assert float(metric(target, preds)) == pytest.approx(
            float(multiclass_brier_score(target, preds, num_classes=3)),
        )
