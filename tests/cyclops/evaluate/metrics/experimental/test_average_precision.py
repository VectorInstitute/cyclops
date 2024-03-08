"""Test average precision metric."""

from functools import partial

import array_api_compat as apc
import array_api_compat.torch
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification import (
    binary_average_precision as tm_binary_average_precision,
)
from torchmetrics.functional.classification import (
    multiclass_average_precision as tm_multiclass_average_precision,
)
from torchmetrics.functional.classification import (
    multilabel_average_precision as tm_multilabel_average_precision,
)

from cyclops.evaluate.metrics.experimental.average_precision import (
    BinaryAveragePrecision,
    MulticlassAveragePrecision,
    MultilabelAveragePrecision,
)
from cyclops.evaluate.metrics.experimental.functional.average_precision import (
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases, _thresholds
from .testers import MetricTester, _inject_ignore_index


def _binary_average_precision_reference(
    target,
    preds,
    thresholds,
    ignore_index,
) -> torch.Tensor:
    """Return the reference binary average precision."""
    return tm_binary_average_precision(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        thresholds=torch.utils.dlpack.from_dlpack(thresholds)
        if apc.is_array_api_obj(thresholds)
        else thresholds,
        ignore_index=ignore_index,
    )


class TestBinaryAveragePrecision(MetricTester):
    """Test binary average precision function and class."""

    atol = 2e-7

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp)[3:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_average_precision_function_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test function for binary average precision using array_api arrays."""
        target, preds = inputs

        if ignore_index is not None:
            if target.shape[1] == 1 and anp.any(target == ignore_index):
                pytest.skip(
                    "When targets are single elements and 'ignore_index' in target "
                    "the function will raise an error because it will receive an "
                    "empty array after filtering out the 'ignore_index' values.",
                )
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_average_precision,
            metric_args={
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_average_precision_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp)[3:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_average_precision_class_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test class for binary average precision using array_api arrays."""
        target, preds = inputs

        if (
            preds.shape[1] == 1
            and is_floating_point(preds)
            and not anp.all(to_int((preds >= 0)) * to_int((preds <= 1)))
        ):
            pytest.skip(
                "When using 0-D logits, batch result will be different from local "
                "result because the `sigmoid` operation may not be applied to each "
                "batch (some values may be in [0, 1] and some may not).",
            )

        if ignore_index is not None:
            if target.shape[1] == 1 and anp.any(target == ignore_index):
                pytest.skip(
                    "When targets are single elements and 'ignore_index' in target "
                    "the function will raise an error because it will receive an "
                    "empty array after filtering out the 'ignore_index' values.",
                )
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=BinaryAveragePrecision,
            metric_args={
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_average_precision_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch)[3:])
    @pytest.mark.parametrize(
        "thresholds",
        _thresholds(xp=array_api_compat.torch),
    )
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_average_precision_with_torch_tensors(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test binary average precision class with torch tensors."""
        target, preds = inputs

        if (
            preds.shape[1] == 1
            and is_floating_point(preds)
            and not torch.all(to_int((preds >= 0)) * to_int((preds <= 1)))
        ):
            pytest.skip(
                "When using 0-D logits, batch result will be different from local "
                "result because the `sigmoid` operation may not be applied to each "
                "batch (some values may be in [0, 1] and some may not).",
            )

        if ignore_index is not None:
            if target.shape[1] == 1 and torch.any(target == ignore_index):
                pytest.skip(
                    "When targets are single elements and 'ignore_index' in target "
                    "the function will raise an error because it will receive an "
                    "empty array after filtering out the 'ignore_index' values.",
                )
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.to(device)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=BinaryAveragePrecision,
            metric_args={
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_average_precision_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_average_precision_reference(
    target,
    preds,
    num_classes=NUM_CLASSES,
    thresholds=None,
    average="macro",
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multiclass average precision."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    return tm_multiclass_average_precision(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_classes,
        average=average,
        thresholds=torch.utils.dlpack.from_dlpack(thresholds)
        if apc.is_array_api_obj(thresholds)
        else thresholds,
        ignore_index=ignore_index,
    )


class TestMulticlassAveragePrecision(MetricTester):
    """Test multiclass average precision function and class."""

    atol = 3e-8

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp)[4:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("average", [None, "none", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multiclass_average_precision_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass average precision."""
        target, preds = inputs

        if ignore_index is not None:
            if target.shape[1] == 1 and anp.any(target == ignore_index):
                pytest.skip(
                    "When targets are single elements and 'ignore_index' in target "
                    "the function will raise an error because it will receive an "
                    "empty array after filtering out the 'ignore_index' values.",
                )
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multiclass_average_precision,
            metric_args={
                "num_classes": NUM_CLASSES,
                "thresholds": thresholds,
                "average": average,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _multiclass_average_precision_reference,
                average=average,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp)[4:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("average", [None, "none", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_average_precision_class_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass average precision."""
        target, preds = inputs

        if ignore_index is not None:
            if target.shape[1] == 1 and anp.any(target == ignore_index):
                pytest.skip(
                    "When targets are single elements and 'ignore_index' in target "
                    "the function will raise an error because it will receive an "
                    "empty array after filtering out the 'ignore_index' values.",
                )
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MulticlassAveragePrecision,
            reference_metric=partial(
                _multiclass_average_precision_reference,
                average=average,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "average": average,
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=array_api_compat.torch)[4:])
    @pytest.mark.parametrize(
        "thresholds",
        _thresholds(xp=array_api_compat.torch),
    )
    @pytest.mark.parametrize("average", [None, "none", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_average_precision_class_with_torch_tensors(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass average precision."""
        target, preds = inputs

        if ignore_index is not None:
            if target.shape[1] == 1 and torch.any(target == ignore_index):
                pytest.skip(
                    "When targets are single elements and 'ignore_index' in target "
                    "the function will raise an error because it will receive an "
                    "empty array after filtering out the 'ignore_index' values.",
                )
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.to(device)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MulticlassAveragePrecision,
            reference_metric=partial(
                _multiclass_average_precision_reference,
                average=average,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "thresholds": thresholds,
                "average": average,
                "ignore_index": ignore_index,
            },
            device=device,
            use_device_for_ref=True,
        )


def _multilabel_average_precision_reference(
    preds,
    target,
    num_labels=NUM_LABELS,
    thresholds=None,
    average="macro",
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multilabel average precision."""
    return tm_multilabel_average_precision(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_labels,
        average=average,
        thresholds=torch.utils.dlpack.from_dlpack(thresholds)
        if apc.is_array_api_obj(thresholds)
        else thresholds,
        ignore_index=ignore_index,
    )


class TestMultilabelAveragePrecision(MetricTester):
    """Test multilabel average precision function and class."""

    atol = 2e-7

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp)[2:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("average", [None, "none", "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_average_precision_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test function for multilabel average precision."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_average_precision,
            reference_metric=partial(
                _multilabel_average_precision_reference,
                average=average,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": thresholds,
                "average": average,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp)[2:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("average", [None, "none", "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_average_precision_class_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel average precision."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelAveragePrecision,
            reference_metric=partial(
                _multilabel_average_precision_reference,
                thresholds=thresholds,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": thresholds,
                "average": average,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=array_api_compat.torch)[2:])
    @pytest.mark.parametrize(
        "thresholds",
        _thresholds(xp=array_api_compat.torch),
    )
    @pytest.mark.parametrize("average", [None, "none", "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_average_precision_class_with_torch_tensors(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel average precision."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.to(device)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelAveragePrecision,
            reference_metric=partial(
                _multilabel_average_precision_reference,
                thresholds=thresholds,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": thresholds,
                "average": average,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
            device=device,
            use_device_for_ref=True,
        )
