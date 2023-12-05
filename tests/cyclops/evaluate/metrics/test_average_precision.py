"""Functions for testing average precision metrics."""


import numpy as np
import pytest
from sklearn.metrics import average_precision_score as sk_average_precision_score

from cyclops.evaluate.metrics.functional import (
    average_precision as cyclops_average_precision,
)
from cyclops.evaluate.metrics.utils import sigmoid

from .helpers import MetricTester
from .inputs import _binary_cases


def _sk_binary_average_precision(
    target: np.ndarray,
    preds: np.ndarray,
    pos_label: int = 1,
) -> float:
    """Compute average precision for binary case using sklearn."""
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)

    return sk_average_precision_score(
        y_true=target,
        y_score=preds,
        pos_label=pos_label,
    )


@pytest.mark.parametrize("inputs", _binary_cases[2:])
class TestBinaryAveragePrecision(MetricTester):
    """Test function and class for computing binary average precision."""

    def test_binary_average_precision_functional(self, inputs):
        """Test function for computing binary average precision."""
        target, preds = inputs

        self.run_functional_test(
            target=target,
            preds=preds,
            metric_functional=cyclops_average_precision,
            sk_metric=_sk_binary_average_precision,
            metric_args={"task": "binary"},
        )
