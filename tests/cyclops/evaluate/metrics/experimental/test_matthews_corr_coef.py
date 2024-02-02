"""Test matthews correlation coefficient metrics."""
from functools import partial

import array_api_compat as apc
import array_api_compat.torch
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification import (
    binary_matthews_corrcoef,
    multiclass_matthews_corrcoef,
    multilabel_matthews_corrcoef,
)

from cyclops.evaluate.metrics.experimental.functional.matthews_corr_coef import (
    binary_mcc,
    multiclass_mcc,
    multilabel_mcc,
)
from cyclops.evaluate.metrics.experimental.matthews_corr_coef import (
    BinaryMCC,
    MulticlassMCC,
    MultilabelMCC,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS, THRESHOLD
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from .testers import MetricTester, _inject_ignore_index


def _binary_mcc_reference(
    target,
    preds,
    threshold,
    ignore_index,
) -> torch.Tensor:
    """Return the reference binary matthews correlation coefficient."""
    return binary_matthews_corrcoef(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        threshold=threshold,
        ignore_index=ignore_index,
    )


class TestBinaryMCC(MetricTester):
    """Test binary matthews correlation coefficient function and class."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_mcc_function_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for binary matthews corrcoef using numpy.array_api arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_mcc,
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_mcc_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_mcc_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for binary matthews correlation coefficient."""
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
            metric_class=BinaryMCC,
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_mcc_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_mcc_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test binary matthews correlation coefficient class with torch tensors."""
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
            metric_class=BinaryMCC,
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _binary_mcc_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_mcc_reference(
    target,
    preds,
    num_classes=NUM_CLASSES,
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multiclass matthews correlation coefficient."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    return multiclass_matthews_corrcoef(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_classes,
        ignore_index=ignore_index,
    )


class TestMulticlassMCC(MetricTester):
    """Test multiclass matthews correlation coefficient function and class."""

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multiclass_mcc_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for multiclass matthews correlation coefficient."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multiclass_mcc,
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
            reference_metric=partial(
                _multiclass_mcc_reference,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_mcc_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for multiclass matthews correlation coefficient."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MulticlassMCC,
            reference_metric=partial(
                _multiclass_mcc_reference,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_mcc_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for multiclass matthews correlation coefficient."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MulticlassMCC,
            reference_metric=partial(
                _multiclass_mcc_reference,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "ignore_index": ignore_index,
            },
            device=device,
            use_device_for_ref=True,
        )


def _multilabel_mcc_reference(
    target,
    preds,
    threshold,
    num_labels=NUM_LABELS,
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multilabel matthews correlation coefficient."""
    return multilabel_matthews_corrcoef(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        num_labels,
        threshold=threshold,
        ignore_index=ignore_index,
    )


class TestMultilabelMCC(MetricTester):
    """Test multilabel matthews correlation coefficient function and class."""

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_mcc_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test function for multilabel matthews correlation coefficient."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_mcc,
            reference_metric=partial(
                _multilabel_mcc_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_mcc_class_with_numpy_array_api_arrays(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for multilabel matthews correlation coefficient."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelMCC,
            reference_metric=partial(
                _multilabel_mcc_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_mcc_class_with_torch_tensors(
        self,
        inputs,
        ignore_index,
    ) -> None:
        """Test class for multilabel matthews correlation coefficient."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelMCC,
            reference_metric=partial(
                _multilabel_mcc_reference,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "num_labels": NUM_LABELS,
                "ignore_index": ignore_index,
            },
        )
