"""Test specificity."""
from functools import partial
from typing import Literal, Optional

import array_api_compat as apc
import array_api_compat.torch
import numpy as np
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification.specificity import (
    binary_specificity as tm_binary_specificity,
)
from torchmetrics.functional.classification.specificity import (
    multiclass_specificity as tm_multiclass_specificity,
)
from torchmetrics.functional.classification.specificity import (
    multilabel_specificity as tm_multilabel_specificity,
)

from cyclops.evaluate.metrics.experimental.functional.specificity import (
    binary_specificity,
    multiclass_specificity,
    multilabel_specificity,
)
from cyclops.evaluate.metrics.experimental.specificity import (
    BinarySpecificity,
    MulticlassSpecificity,
    MultilabelSpecificity,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS, THRESHOLD
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from .testers import MetricTester, _inject_ignore_index


def _binary_specificity_reference(
    target,
    preds,
    threshold,
    ignore_index,
) -> torch.Tensor:
    """Compute binary specificity using torchmetrics."""
    return tm_binary_specificity(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        threshold=threshold,
        ignore_index=ignore_index,
    )


class TestBinarySpecificity(MetricTester):
    """Test binary specificity metric class and function."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_specificity_function_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for binary specificity using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_specificity,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_specificity_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_specificity_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for binary specificity using `numpy.array_api` arrays."""
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
            metric_class=BinarySpecificity,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_specificity_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_specificity_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test binary specificity class with torch tensors."""
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
            metric_class=BinarySpecificity,
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_specificity_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_specificity_reference(
    target,
    preds,
    num_classes=NUM_CLASSES,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted"]] = "micro",
    ignore_index=None,
) -> torch.Tensor:
    """Compute multiclass specificity using torchmetrics."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    return tm_multiclass_specificity(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_classes=num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )


class TestMulticlassSpecificity(MetricTester):
    """Test multiclass specificity metric class and function."""

    atol = 2e-7

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_specificity_function_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass specificity using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                multiclass_specificity(
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
                metric_function=multiclass_specificity,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_specificity_reference,
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
    def test_multiclass_specificity_class_with_numpy_array_api_arrays(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass specificity using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassSpecificity(
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
                metric_class=MulticlassSpecificity,
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_specificity_reference,
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
    def test_multiclass_specificity_class_with_torch_tensors(
        self,
        inputs,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test multiclass specificity class with torch tensors."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassSpecificity(
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
                metric_class=MulticlassSpecificity,
                reference_metric=partial(
                    _multiclass_specificity_reference,
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


def _multilabel_specificity_reference(
    target,
    preds,
    threshold,
    num_labels=NUM_LABELS,
    average: Optional[Literal["micro", "macro", "weighted"]] = "macro",
    ignore_index=None,
) -> torch.Tensor:
    """Compute multilabel specificity using torchmetrics."""
    return tm_multilabel_specificity(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_labels=num_labels,
        threshold=threshold,
        average=average,
        ignore_index=ignore_index,
    )


class TestMultilabelSpecificity(MetricTester):
    """Test multilabel specificity function and class."""

    atol = 2e-7

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_specificity_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test function for multilabel specificity with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_specificity,
            reference_metric=partial(
                _multilabel_specificity_reference,
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
    def test_multilabel_specificity_class_with_numpy_array_api_arrays(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel specificity with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelSpecificity,
            reference_metric=partial(
                _multilabel_specificity_reference,
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
    def test_multilabel_specificity_class_with_torch_tensors(
        self,
        inputs,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel specificity with torch tensors."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelSpecificity,
            reference_metric=partial(
                _multilabel_specificity_reference,
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


def test_top_k_multilabel_specificity():
    """Test top-k multilabel specificity."""
    target = anp.asarray([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = anp.asarray([[0.1, 0.9, 0.8, 0.3], [0.9, 0.1, 0.8, 0.3]])
    expected_result = anp.asarray([1.0, 1.0, 0.0, 1.0], dtype=anp.float32)

    result = multilabel_specificity(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    metric = MultilabelSpecificity(num_labels=4, average=None, top_k=2)
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
    expected_result = anp.asarray([0.0, 0.0, 0.5, 0.5], dtype=anp.float32)

    result = multilabel_specificity(target, preds, num_labels=4, average=None, top_k=2)
    assert np.allclose(result, expected_result)

    class_result = metric(target, preds)
    assert np.allclose(class_result, expected_result)
