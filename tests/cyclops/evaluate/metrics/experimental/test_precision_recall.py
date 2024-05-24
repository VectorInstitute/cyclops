"""Test precision recall metrics."""

from functools import partial
from typing import Literal, Optional

import array_api_compat as apc
import array_api_compat.torch
import numpy as np
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification.precision_recall import (
    binary_precision as tm_binary_precision,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_recall as tm_binary_recall,
)
from torchmetrics.functional.classification.precision_recall import (
    multiclass_precision as tm_multiclass_precision,
)
from torchmetrics.functional.classification.precision_recall import (
    multiclass_recall as tm_multiclass_recall,
)
from torchmetrics.functional.classification.precision_recall import (
    multilabel_precision as tm_multilabel_precision,
)
from torchmetrics.functional.classification.precision_recall import (
    multilabel_recall as tm_multilabel_recall,
)

from cyclops.evaluate.metrics.experimental.functional.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
)
from cyclops.evaluate.metrics.experimental.precision_recall import (
    BinaryPrecision,
    BinaryRecall,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS, THRESHOLD
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from .testers import MetricTester, _inject_ignore_index


def _binary_precision_recall_reference(
    metric_name: Literal["precision", "recall"],
    target,
    preds,
    threshold,
    ignore_index,
) -> torch.Tensor:
    if metric_name == "precision":
        return tm_binary_precision(
            torch.utils.dlpack.from_dlpack(preds),
            torch.utils.dlpack.from_dlpack(target),
            threshold=threshold,
            ignore_index=ignore_index,
        )

    return tm_binary_recall(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        threshold=threshold,
        ignore_index=ignore_index,
    )


class TestBinaryPrecision(MetricTester):
    """Test binary precision metric class and function."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_precision_function_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for binary precision using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_precision,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_precision_recall_reference,
                metric_name="precision",
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_precision_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for binary precision using `numpy.array_api` arrays."""
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
            metric_class=BinaryPrecision,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_precision_recall_reference,
                metric_name="precision",
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_precision_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test binary precision class with torch tensors."""
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
            metric_class=BinaryPrecision,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_precision_recall_reference,
                metric_name="precision",
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


class TestBinaryRecall(MetricTester):
    """Test binary recall metric class and function."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_recall_function_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for binary recall using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_recall,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_precision_recall_reference,
                metric_name="recall",
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_recall_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for binary recall using `numpy.array_api` arrays."""
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
            metric_class=BinaryRecall,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_precision_recall_reference,
                metric_name="recall",
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_recall_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test binary recall class with torch tensors."""
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
            metric_class=BinaryRecall,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_precision_recall_reference,
                metric_name="recall",
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_precision_recall_reference(
    metric_name: Literal["precision", "recall"],
    target,
    preds,
    num_classes=NUM_CLASSES,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted"]] = "micro",
    ignore_index=None,
) -> torch.Tensor:
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    if metric_name == "precision":
        return tm_multiclass_precision(
            torch.utils.dlpack.from_dlpack(preds),
            torch.utils.dlpack.from_dlpack(target),
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
        )

    return tm_multiclass_recall(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_classes=num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )


class TestMulticlassPrecision(MetricTester):
    """Test multiclass precision metric class and function."""

    atol = 6e-8

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_precision_function_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass precision using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                multiclass_precision(
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
                metric_function=multiclass_precision,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_precision_recall_reference,
                    metric_name="precision",
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
    def test_multiclass_precision_class_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass precision using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassPrecision(
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
                metric_class=MulticlassPrecision,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_precision_recall_reference,
                    metric_name="precision",
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
    def test_multiclass_precision_class_with_torch_tensors(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test multiclass precision class with torch tensors."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassPrecision(
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
                metric_class=MulticlassPrecision,
                reference_metric=partial(
                    _multiclass_precision_recall_reference,
                    metric_name="precision",
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


class TestMulticlassRecall(MetricTester):
    """Test multiclass recall metric class and function."""

    atol = 3e-8

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_recall_function_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass recall using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                multiclass_recall(
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
                metric_function=multiclass_recall,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_precision_recall_reference,
                    metric_name="recall",
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
    def test_multiclass_recall_class_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass recall using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassRecall(
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
                metric_class=MulticlassRecall,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_precision_recall_reference,
                    metric_name="recall",
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
    def test_multiclass_recall_class_with_torch_tensors(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test multiclass recall class with torch tensors."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassRecall(
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
                metric_class=MulticlassRecall,
                reference_metric=partial(
                    _multiclass_precision_recall_reference,
                    metric_name="recall",
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


def _multilabel_precision_recall_reference(
    metric_name: Literal["precision", "recall"],
    target,
    preds,
    threshold,
    num_labels=NUM_LABELS,
    average: Optional[Literal["micro", "macro", "weighted"]] = "macro",
    ignore_index=None,
) -> torch.Tensor:
    if metric_name == "precision":
        return tm_multilabel_precision(
            torch.utils.dlpack.from_dlpack(preds),
            torch.utils.dlpack.from_dlpack(target),
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            ignore_index=ignore_index,
        )

    return tm_multilabel_recall(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_labels=num_labels,
        threshold=threshold,
        average=average,
        ignore_index=ignore_index,
    )


class TestMultilabelPrecision(MetricTester):
    """Test multilabel precision function and class."""

    atol = 6e-8

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_precision_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test function for multilabel precision with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_precision,
            reference_metric=partial(
                _multilabel_precision_recall_reference,
                metric_name="precision",
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
    def test_multilabel_precision_class_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel precision with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelPrecision,
            reference_metric=partial(
                _multilabel_precision_recall_reference,
                metric_name="precision",
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
    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_precision_class_with_torch_tensors(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel precision with torch tensors."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelPrecision,
            reference_metric=partial(
                _multilabel_precision_recall_reference,
                metric_name="precision",
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
            device=device,
            use_device_for_ref=True,
        )


def test_top_k_multilabel_precision():
    """Test top-k multilabel precision."""
    target = anp.asarray([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = anp.asarray([[0.1, 0.9, 0.8, 0.3], [0.9, 0.1, 0.8, 0.3]])
    expected_result = anp.asarray([1.0, 1.0, 1.0, 0.0], dtype=anp.float32)

    result = multilabel_precision(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    metric = MultilabelPrecision(num_labels=4, average=None, top_k=2)
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
    expected_result = anp.asarray([0.25, 0.0, 0.0, 0.0], dtype=anp.float32)

    result = multilabel_precision(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    class_result = metric(target, preds)
    assert np.allclose(class_result, expected_result)


class TestMultilabelRecall(MetricTester):
    """Test multilabel recall function and class."""

    atol = 6e-8

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_recall_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test function for multilabel recall with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_recall,
            reference_metric=partial(
                _multilabel_precision_recall_reference,
                metric_name="recall",
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
    def test_multilabel_recall_class_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel recall with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelRecall,
            reference_metric=partial(
                _multilabel_precision_recall_reference,
                metric_name="recall",
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
    def test_multilabel_recall_class_with_torch_tensors(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel recall with torch tensors."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelRecall,
            reference_metric=partial(
                _multilabel_precision_recall_reference,
                metric_name="recall",
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


def test_top_k_multilabel_recall():
    """Test top-k multilabel recall."""
    target = anp.asarray([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = anp.asarray([[0.1, 0.9, 0.8, 0.3], [0.9, 0.1, 0.8, 0.3]])
    expected_result = anp.asarray([1.0, 1.0, 1.0, 0.0], dtype=anp.float32)

    result = multilabel_recall(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    metric = MultilabelRecall(num_labels=4, average=None, top_k=2)
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
    expected_result = anp.asarray([1.0, 0.0, 0.0, 0.0], dtype=anp.float32)

    result = multilabel_recall(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    class_result = metric(target, preds)
    assert np.allclose(class_result, expected_result)
