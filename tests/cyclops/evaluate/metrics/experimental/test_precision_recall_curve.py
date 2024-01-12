"""Test precision-recall curve metric."""
from functools import partial
from typing import List, Tuple, Union

import array_api_compat as apc
import array_api_compat.torch
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification import (
    binary_precision_recall_curve as tm_binary_precision_recall_curve,
)
from torchmetrics.functional.classification import (
    multiclass_precision_recall_curve as tm_multiclass_precision_recall_curve,
)
from torchmetrics.functional.classification import (
    multilabel_precision_recall_curve as tm_multilabel_precision_recall_curve,
)

from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from cyclops.evaluate.metrics.experimental.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases, _thresholds
from .testers import MetricTester, _inject_ignore_index


def _binary_precision_recall_curve_reference(
    target,
    preds,
    thresholds,
    ignore_index,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the reference binary precision-recall curve."""
    return tm_binary_precision_recall_curve(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        thresholds=torch.utils.dlpack.from_dlpack(thresholds)
        if apc.is_array_api_obj(thresholds)
        else thresholds,
        ignore_index=ignore_index,
    )


class TestBinaryPrecisionRecallCurve(MetricTester):
    """Test binary precision-recall curve function and class."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp)[3:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_precision_recall_curve_function_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test function for binary precision-recall curve using array_api arrays."""
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
            metric_function=binary_precision_recall_curve,
            metric_args={
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_precision_recall_curve_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp)[3:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_precision_recall_curve_class_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test class for binary precision-recall curve using array_api arrays."""
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
            metric_class=BinaryPrecisionRecallCurve,
            metric_args={
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_precision_recall_curve_reference,
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
    def test_binary_precision_recall_curve_with_torch_tensors(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test binary precision-recall curve class with torch tensors."""
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
            metric_class=BinaryPrecisionRecallCurve,
            metric_args={
                "thresholds": thresholds,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_precision_recall_curve_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_precision_recall_curve_reference(
    target,
    preds,
    num_classes=NUM_CLASSES,
    thresholds=None,
    ignore_index=None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
]:
    """Return the reference multiclass precision-recall curve."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    return tm_multiclass_precision_recall_curve(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_classes,
        thresholds=torch.utils.dlpack.from_dlpack(thresholds)
        if apc.is_array_api_obj(thresholds)
        else thresholds,
        ignore_index=ignore_index,
    )


class TestMulticlassPrecisionRecallCurve(MetricTester):
    """Test multiclass precision-recall curve function and class."""

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp)[4:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("average", [None, "none"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multiclass_precision_recall_curve_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass precision-recall curve."""
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
            metric_function=multiclass_precision_recall_curve,
            metric_args={
                "num_classes": NUM_CLASSES,
                "thresholds": thresholds,
                "average": average,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _multiclass_precision_recall_curve_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp)[4:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("average", [None, "none"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_precision_recall_curve_class_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass precision-recall curve."""
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
            metric_class=MulticlassPrecisionRecallCurve,
            reference_metric=partial(
                _multiclass_precision_recall_curve_reference,
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
    @pytest.mark.parametrize("average", [None, "none"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_precision_recall_curve_class_with_torch_tensors(
        self,
        inputs,
        thresholds,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass precision-recall curve."""
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
            metric_class=MulticlassPrecisionRecallCurve,
            reference_metric=partial(
                _multiclass_precision_recall_curve_reference,
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


def _multilabel_precision_recall_curve_reference(
    preds,
    target,
    num_labels=NUM_LABELS,
    thresholds=None,
    ignore_index=None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]],
]:
    """Return the reference multilabel precision-recall curve."""
    return tm_multilabel_precision_recall_curve(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_labels,
        thresholds=torch.utils.dlpack.from_dlpack(thresholds)
        if apc.is_array_api_obj(thresholds)
        else thresholds,
        ignore_index=ignore_index,
    )


class TestMultilabelPrecisionRecallCurve(MetricTester):
    """Test multilabel precision-recall curve function and class."""

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp)[2:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_precision_recall_curve_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test function for multilabel precision-recall curve."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_precision_recall_curve,
            reference_metric=partial(
                _multilabel_precision_recall_curve_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": thresholds,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp)[2:])
    @pytest.mark.parametrize("thresholds", _thresholds(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_precision_recall_curve_class_with_numpy_array_api_arrays(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test class for multilabel precision-recall curve."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelPrecisionRecallCurve,
            reference_metric=partial(
                _multilabel_precision_recall_curve_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": thresholds,
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
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_precision_recall_curve_class_with_torch_tensors(
        self,
        inputs,
        thresholds,
        ignore_index,
    ) -> None:
        """Test class for multilabel precision-recall curve."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(thresholds, torch.Tensor):
            thresholds = thresholds.to(device)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelPrecisionRecallCurve,
            reference_metric=partial(
                _multilabel_precision_recall_curve_reference,
                thresholds=thresholds,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": thresholds,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
            device=device,
            use_device_for_ref=True,
        )
