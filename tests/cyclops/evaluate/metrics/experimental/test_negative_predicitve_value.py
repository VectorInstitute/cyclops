"""Test negative predictive value."""
from functools import partial
from typing import Literal, Optional

import array_api_compat as apc
import array_api_compat.torch
import numpy as np
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torch import Tensor
from torchmetrics.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide

from cyclops.evaluate.metrics.experimental.functional.negative_predictive_value import (
    binary_npv,
    multiclass_npv,
    multilabel_npv,
)
from cyclops.evaluate.metrics.experimental.negative_predictive_value import (
    BinaryNPV,
    MulticlassNPV,
    MultilabelNPV,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS, THRESHOLD
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from .testers import MetricTester, _inject_ignore_index


def _npv_reduce(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
    multilabel: bool = False,
) -> Tensor:
    if average == "binary":
        return _safe_divide(tn, tn + fn)
    if average == "micro":
        tn = tn.sum(dim=0)
        fn = fn.sum(dim=0)
        return _safe_divide(tn, tn + fn)

    npv_score = _safe_divide(tn, tn + fn)
    return _adjust_weights_safe_divide(npv_score, average, multilabel, tp, fp, fn)


def _binary_npv_reference(
    target,
    preds,
    threshold,
    ignore_index,
) -> torch.Tensor:
    """Compute binary negative predictive value using torchmetrics."""
    preds = torch.utils.dlpack.from_dlpack(preds)
    target = torch.utils.dlpack.from_dlpack(target)
    _binary_stat_scores_arg_validation(threshold, ignore_index=ignore_index)
    _binary_stat_scores_tensor_validation(preds, target, ignore_index=ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target)
    return _npv_reduce(tp, fp, tn, fn, average="binary")


class TestBinaryNPV(MetricTester):
    """Test binary negative predictive value metric class and function."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_npv_function_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for binary NPV using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_npv,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_npv_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_npv_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for binary NPV using `numpy.array_api` arrays."""
        target, preds = inputs

        if (
            preds.ndim == 1
            and is_floating_point(preds)
            and not anp.all(to_int((preds >= 0)) * to_int((preds <= 1)))
        ):
            pytest.skip(
                "When using 0-D logits, batch result will be different from local "
                "result because the `sigmoid` operation may not be applied to each "
                "batch (some values may be in [0, 1] and some may not).",
            )

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=BinaryNPV,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_npv_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_npv_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test binary negative predictive value class with torch tensors."""
        target, preds = inputs

        if (
            preds.ndim == 1
            and is_floating_point(preds)
            and not torch.all(to_int((preds >= 0)) * to_int((preds <= 1)))
        ):
            pytest.skip(
                "When using 0-D logits, batch result will be different from local "
                "result because the `sigmoid` operation may not be applied to each "
                "batch (some values may be in [0, 1] and some may not).",
            )

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=BinaryNPV,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_npv_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_npv_reference(
    target,
    preds,
    num_classes=NUM_CLASSES,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted"]] = "micro",
    ignore_index=None,
) -> torch.Tensor:
    """Compute multiclass negative predictive value using torchmetrics."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    preds = torch.utils.dlpack.from_dlpack(preds)
    target = torch.utils.dlpack.from_dlpack(target)
    _multiclass_stat_scores_arg_validation(
        num_classes,
        top_k,
        average,
        ignore_index=ignore_index,
    )
    _multiclass_stat_scores_tensor_validation(
        preds,
        target,
        num_classes,
        ignore_index=ignore_index,
    )
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds,
        target,
        num_classes,
        top_k,
        average,
        ignore_index=ignore_index,
    )
    return _npv_reduce(tp, fp, tn, fn, average=average)


class TestMulticlassNPV(MetricTester):
    """Test multiclass negative predictive value metric class and function."""

    atol = 2e-7

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_npv_function_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass NPV using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                multiclass_npv(
                    target,
                    preds,
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                )
        else:
            self.run_metric_function_implementation_test(
                target,
                preds,
                metric_function=multiclass_npv,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_npv_reference,
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                ),
            )

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_npv_class_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass NPV using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassNPV(
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                )
                metric(target, preds)
        else:
            self.run_metric_class_implementation_test(
                target,
                preds,
                metric_class=MulticlassNPV,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_npv_reference,
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                ),
            )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_npv_class_with_torch_tensors(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test multiclass negative predictive value class with torch tensors."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassNPV(
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                )
                metric(target, preds)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.run_metric_class_implementation_test(
                target,
                preds,
                metric_class=MulticlassNPV,
                reference_metric=partial(
                    _multiclass_npv_reference,
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                ),
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                device=device,
                use_device_for_ref=True,
            )


def _multilabel_npv_reference(
    target,
    preds,
    threshold,
    num_labels=NUM_LABELS,
    average: Optional[Literal["micro", "macro", "weighted"]] = "macro",
    ignore_index=None,
) -> torch.Tensor:
    """Compute multilabel negative predictive value using torchmetrics."""
    preds = torch.utils.dlpack.from_dlpack(preds)
    target = torch.utils.dlpack.from_dlpack(target)
    _multilabel_stat_scores_arg_validation(
        num_labels,
        threshold,
        average,
        ignore_index=ignore_index,
    )
    _multilabel_stat_scores_tensor_validation(
        preds,
        target,
        num_labels,
        "global",
        ignore_index=ignore_index,
    )
    preds, target = _multilabel_stat_scores_format(
        preds,
        target,
        num_labels,
        threshold,
        ignore_index=ignore_index,
    )
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target)
    return _npv_reduce(tp, fp, tn, fn, average=average, multilabel=True)


class TestMultilabelNPV(MetricTester):
    """Test multilabel negative predictive value function and class."""

    atol = 6e-8

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_npv_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test function for multilabel NPV with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_npv,
            reference_metric=partial(
                _multilabel_npv_reference,
                num_labels=NUM_LABELS,
                threshold=THRESHOLD,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_npv_class_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel NPV with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelNPV,
            reference_metric=partial(
                _multilabel_npv_reference,
                num_labels=NUM_LABELS,
                threshold=THRESHOLD,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_npv_class_with_torch_tensors(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel negative predictive value with torch tensors."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelNPV,
            reference_metric=partial(
                _multilabel_npv_reference,
                num_labels=NUM_LABELS,
                threshold=THRESHOLD,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "average": average,
                "ignore_index": ignore_index,
            },
        )


def test_top_k_multilabel_npv():
    """Test top-k multilabel negative predictive value."""
    target = anp.asarray([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = anp.asarray([[0.1, 0.9, 0.8, 0.3], [0.9, 0.1, 0.8, 0.3]])
    expected_result = anp.asarray([1.0, 1.0, 0.0, 1.0], dtype=anp.float32)

    result = multilabel_npv(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    metric = MultilabelNPV(num_labels=4, average=None, top_k=2)
    metric(target, preds)
    class_result = metric.compute()
    assert np.allclose(class_result, expected_result)
    metric.reset()

    preds = anp.asarray(
        [
            [[0.57, 0.63], [0.33, 0.55], [0.73, 0.55], [0.36, 0.66]],
            [[0.78, 0.94], [0.47, 0.31], [0.14, 0.28], [0.35, 0.81]],
        ],
    )
    target = anp.asarray(
        [[[0, 0], [1, 1], [0, 1], [0, 0]], [[0, 1], [0, 1], [1, 0], [0, 0]]],
    )
    expected_result = anp.asarray([0.0, 0.0, 0.33333334, 1.0], dtype=anp.float32)

    result = multilabel_npv(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    class_result = metric(target, preds)
    assert np.allclose(class_result, expected_result)
