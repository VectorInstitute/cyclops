"""Test confusion matrix metrics."""
from functools import partial

import array_api_compat as apc
import array_api_compat.torch
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification import (
    binary_confusion_matrix as tm_binary_confusion_matrix,
)
from torchmetrics.functional.classification import (
    multiclass_confusion_matrix as tm_multiclass_confusion_matrix,
)
from torchmetrics.functional.classification import (
    multilabel_confusion_matrix as tm_multilabel_confusion_matrix,
)

from cyclops.evaluate.metrics.experimental.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from cyclops.evaluate.metrics.experimental.functional.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS, THRESHOLD
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from .testers import MetricTester, _inject_ignore_index


def _binary_confusion_matrix_reference(
    target,
    preds,
    threshold,
    normalize,
    ignore_index,
) -> torch.Tensor:
    """Return the reference binary confusion matrix."""
    return tm_binary_confusion_matrix(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        threshold=threshold,
        normalize=normalize,
        ignore_index=ignore_index,
    )


class TestBinaryConfusionMatrix(MetricTester):
    """Test binary confusion matrix function and class."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_confusion_matrix_function_with_numpy_array_api_arrays(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test function for binary confusion matrix using numpy.array_api arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_confusion_matrix,
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_confusion_matrix_reference,
                threshold=THRESHOLD,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_confusion_matrix_class_with_numpy_array_api_arrays(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test class for binary confusion matrix."""
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
            metric_class=BinaryConfusionMatrix,
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_confusion_matrix_reference,
                threshold=THRESHOLD,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_confusion_matrix_class_with_torch_tensors(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test binary confusion matrix class with torch tensors."""
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
            metric_class=BinaryConfusionMatrix,
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_confusion_matrix_reference,
                threshold=THRESHOLD,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_confusion_matrix_reference(
    target,
    preds,
    num_classes=NUM_CLASSES,
    normalize=None,
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multiclass confusion matrix."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    return tm_multiclass_confusion_matrix(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_classes,
        normalize=normalize,
        ignore_index=ignore_index,
    )


class TestMulticlassConfusionMatrix(MetricTester):
    """Test multiclass confusion matrix function and class."""

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multiclass_confusion_matrix_with_numpy_array_api_arrays(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test function for multiclass confusion matrix."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multiclass_confusion_matrix,
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _multiclass_confusion_matrix_reference,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_confusion_matrix_class_with_numpy_array_api_arrays(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test class for multiclass confusion matrix."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MulticlassConfusionMatrix,
            reference_metric=partial(
                _multiclass_confusion_matrix_reference,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_confusion_matrix_class_with_torch_tensors(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test class for multiclass confusion matrix."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MulticlassConfusionMatrix,
            reference_metric=partial(
                _multiclass_confusion_matrix_reference,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
            device=device,
            use_device_for_ref=True,
        )


def _multilabel_confusion_matrix_reference(
    preds,
    target,
    threshold,
    num_labels=NUM_LABELS,
    normalize=None,
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multilabel confusion matrix."""
    return tm_multilabel_confusion_matrix(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_labels,
        threshold=threshold,
        normalize=normalize,
        ignore_index=ignore_index,
    )


class TestMultilabelConfusionMatrix(MetricTester):
    """Test multilabel confusion matrix function and class."""

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_confusion_matrix_with_numpy_array_api_arrays(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test function for multilabel confusion matrix."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_confusion_matrix,
            reference_metric=partial(
                _multilabel_confusion_matrix_reference,
                threshold=THRESHOLD,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_confusion_matrix_class_with_numpy_array_api_arrays(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test class for multilabel confusion matrix."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelConfusionMatrix,
            reference_metric=partial(
                _multilabel_confusion_matrix_reference,
                threshold=THRESHOLD,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_confusion_matrix_class_with_torch_tensors(
        self,
        inputs,
        normalize,
        ignore_index,
    ) -> None:
        """Test class for multilabel confusion matrix."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelConfusionMatrix,
            reference_metric=partial(
                _multilabel_confusion_matrix_reference,
                threshold=THRESHOLD,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )
