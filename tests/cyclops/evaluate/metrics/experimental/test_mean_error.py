"""Tests for mean error metrics."""

from functools import partial

import array_api_compat.torch
import numpy.array_api as anp
import pytest
import torch
import torch.utils.dlpack
from array_api_compat.common._helpers import _is_torch_array
from torchmetrics.functional import (
    mean_absolute_error as tm_mean_absolute_error,
)
from torchmetrics.functional import (
    mean_absolute_percentage_error as tm_mean_abs_percentage_error,
)
from torchmetrics.functional import (
    mean_squared_error as tm_mean_squared_error,
)
from torchmetrics.functional import (
    symmetric_mean_absolute_percentage_error as tm_smape,
)
from torchmetrics.functional import (
    weighted_mean_absolute_percentage_error as tm_wmape,
)

from cyclops.evaluate.metrics.experimental import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
    WeightedMeanAbsolutePercentageError,
)
from cyclops.evaluate.metrics.experimental.functional import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_mean_absolute_percentage_error,
)

from .inputs import NUM_LABELS, _regression_cases
from .testers import MetricTester


def _tm_metric_wrapper(target, preds, tm_fn, metric_args) -> torch.Tensor:
    target = torch.utils.dlpack.from_dlpack(target)
    preds = torch.utils.dlpack.from_dlpack(preds)
    return tm_fn(preds, target, **metric_args)


@pytest.mark.parametrize(
    "inputs",
    (*_regression_cases(xp=anp), *_regression_cases(xp=array_api_compat.torch)),
)
@pytest.mark.parametrize(
    "metric_class, metric_functional, tm_fn, metric_args",
    [
        (MeanAbsoluteError, mean_absolute_error, tm_mean_absolute_error, {}),
        (
            MeanSquaredError,
            mean_squared_error,
            tm_mean_squared_error,
            {"squared": True},
        ),
        (
            MeanSquaredError,
            mean_squared_error,
            tm_mean_squared_error,
            {"squared": False},
        ),
        (
            MeanSquaredError,
            mean_squared_error,
            tm_mean_squared_error,
            {"squared": True, "num_outputs": NUM_LABELS},
        ),
        (
            MeanAbsolutePercentageError,
            mean_absolute_percentage_error,
            tm_mean_abs_percentage_error,
            {},
        ),
        (
            SymmetricMeanAbsolutePercentageError,
            symmetric_mean_absolute_percentage_error,
            tm_smape,
            {},
        ),
        (
            WeightedMeanAbsolutePercentageError,
            weighted_mean_absolute_percentage_error,
            tm_wmape,
            {},
        ),
    ],
)
class TestMeanError(MetricTester):
    """Test class for `MeanError` metric."""

    atol = 2e-6

    def test_mean_error_class(
        self,
        inputs,
        metric_class,
        metric_functional,
        tm_fn,
        metric_args,
    ):
        """Test class implementation of metric."""
        target, preds = inputs
        device = "cpu"
        if _is_torch_array(target) and torch.cuda.is_available():
            device = "cuda"

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=metric_class,
            reference_metric=partial(
                _tm_metric_wrapper,
                tm_fn=tm_fn,
                metric_args=metric_args,
            ),
            metric_args=metric_args,
            device=device,
            use_device_for_ref=_is_torch_array(target),
        )

    def test_mean_error_functional(
        self,
        inputs,
        metric_class,
        metric_functional,
        tm_fn,
        metric_args,
    ):
        """Test functional implementation of metric."""
        target, preds = inputs
        device = "cpu"
        if _is_torch_array(target) and torch.cuda.is_available():
            device = "cuda"

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=metric_functional,
            reference_metric=partial(
                _tm_metric_wrapper,
                tm_fn=tm_fn,
                metric_args=metric_args,
            ),
            metric_args=metric_args,
            device=device,
            use_device_for_ref=_is_torch_array(target),
        )


@pytest.mark.parametrize(
    "metric_class",
    [
        MeanSquaredError,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        WeightedMeanAbsolutePercentageError,
        SymmetricMeanAbsolutePercentageError,
    ],
)
def test_error_on_different_shape(metric_class):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        ValueError,
        match="Expected `target` and `preds` to have the same shape, but got `target`.*",
    ):
        metric(torch.randn(100), torch.randn(50))
