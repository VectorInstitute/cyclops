"""Tests for the F-score metric."""
from functools import partial
from typing import Literal, Optional

import array_api_compat as apc
import array_api_compat.torch
import numpy as np
import numpy.array_api as anp
import pytest
import torch.utils.dlpack
from torchmetrics.functional.classification.f_beta import (
    binary_fbeta_score as tm_binary_fbeta_score,
)
from torchmetrics.functional.classification.f_beta import (
    multiclass_fbeta_score as tm_multiclass_fbeta_score,
)
from torchmetrics.functional.classification.f_beta import (
    multilabel_fbeta_score as tm_multilabel_fbeta_score,
)

from cyclops.evaluate.metrics.experimental.f_score import (
    BinaryF1Score,
    BinaryFBetaScore,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MultilabelF1Score,
    MultilabelFBetaScore,
)
from cyclops.evaluate.metrics.experimental.functional.f_score import (
    binary_f1_score,
    binary_fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from cyclops.evaluate.metrics.experimental.utils.ops import to_int
from cyclops.evaluate.metrics.experimental.utils.validation import is_floating_point

from ..conftest import NUM_CLASSES, NUM_LABELS, THRESHOLD
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from .testers import MetricTester, _inject_ignore_index


def _binary_fbeta_score_reference(
    target: torch.Tensor,
    preds: torch.Tensor,
    beta: float,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
):
    return tm_binary_fbeta_score(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        beta=beta,
        threshold=threshold,
        ignore_index=ignore_index,
    )


@pytest.mark.parametrize("beta", [1.0, 2.0])
class TestBinaryFBetaScore(MetricTester):
    """Test binary F-beta function and class."""

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_fbeta_score_function_with_numpy_array_api_arrays(
        self,
        inputs,
        beta,
        ignore_index,
    ) -> None:
        """Test binary f-beta function using `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=binary_f1_score
            if beta == 1.0
            else partial(binary_fbeta_score, beta=beta),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_fbeta_score_reference,
                beta=beta,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.parametrize("inputs", _binary_cases(xp=anp))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_fbeta_score_class_with_numpy_array_api_arrays(
        self,
        inputs,
        beta,
        ignore_index,
    ) -> None:
        """Test class for binary fbeta using `numpy.array_api` arrays."""
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
            metric_class=BinaryF1Score
            if beta == 1.0
            else partial(BinaryFBetaScore, beta=beta),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_fbeta_score_reference,
                beta=beta,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
        )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _binary_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_fbeta_class_with_torch_tensors(
        self,
        inputs,
        beta,
        ignore_index,
    ) -> None:
        """Test binary fbeta class with torch tensors."""
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
            metric_class=BinaryF1Score
            if beta == 1.0
            else partial(BinaryFBetaScore, beta=beta),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index},
            reference_metric=partial(
                _binary_fbeta_score_reference,
                beta=beta,
                threshold=THRESHOLD,
                ignore_index=ignore_index,
            ),
            device=device,
            use_device_for_ref=True,
        )


def _multiclass_fbeta_score_reference(
    target,
    preds,
    beta,
    num_classes=NUM_CLASSES,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted"]] = "macro",
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multiclass f-beta score."""
    if preds.ndim == 1 and is_floating_point(preds):
        xp = apc.array_namespace(preds)
        preds = xp.argmax(preds, axis=0)

    return tm_multiclass_fbeta_score(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        beta=beta,
        num_classes=num_classes,
        top_k=top_k,
        average=average,
        ignore_index=ignore_index,
    )


@pytest.mark.parametrize("beta", [1.0, 2.0])
class TestMulticlassFBetaScore(MetricTester):
    """Test multiclass F-beta function and class."""

    atol = 3e-8

    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=anp))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_fbeta_score_with_numpy_array_api_arrays(
        self,
        inputs,
        beta,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test function for multiclass fbeta score with `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                multiclass_fbeta_score(
                    target,
                    preds,
                    beta,
                    num_classes=NUM_CLASSES,
                    top_k=top_k,
                    average=average,
                    ignore_index=ignore_index,
                )
        else:
            self.run_metric_function_implementation_test(
                target,
                preds,
                metric_function=multiclass_f1_score
                if beta == 1.0
                else partial(multiclass_fbeta_score, beta=beta),
                metric_args={
                    "num_classes": NUM_CLASSES,
                    "top_k": top_k,
                    "average": average,
                    "ignore_index": ignore_index,
                },
                reference_metric=partial(
                    _multiclass_fbeta_score_reference,
                    beta=beta,
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
    def test_multiclass_fbeta_score_class_with_numpy_array_api_arrays(
        self,
        inputs,
        beta,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass fbeta score with `numpy.array_api` arrays."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassFBetaScore(
                    beta=beta,
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
                metric_class=MulticlassF1Score
                if beta == 1.0
                else partial(MulticlassFBetaScore, beta=beta),
                reference_metric=partial(
                    _multiclass_fbeta_score_reference,
                    beta=beta,
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
            )

    @pytest.mark.integration_test()  # machine for integration tests has GPU
    @pytest.mark.parametrize("inputs", _multiclass_cases(xp=array_api_compat.torch))
    @pytest.mark.parametrize("top_k", [1, 2])
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 1, -1])
    def test_multiclass_fbeta_score_class_with_torch_tensors(
        self,
        inputs,
        beta,
        top_k,
        average,
        ignore_index,
    ) -> None:
        """Test class for multiclass fbeta score with torch tensors."""
        target, preds = inputs

        if ignore_index is not None:
            target = _inject_ignore_index(target, ignore_index)

        if top_k > 1 and not is_floating_point(preds):
            with pytest.raises(ValueError):
                metric = MulticlassFBetaScore(
                    beta=beta,
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
                metric_class=MulticlassF1Score
                if beta == 1.0
                else partial(MulticlassFBetaScore, beta=beta),
                reference_metric=partial(
                    _multiclass_fbeta_score_reference,
                    beta=beta,
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


def _multilabel_fbeta_score_reference(
    target,
    preds,
    beta,
    threshold,
    num_labels=NUM_LABELS,
    average: Optional[Literal["micro", "macro", "weighted"]] = "macro",
    ignore_index=None,
) -> torch.Tensor:
    """Return the reference multilabel f-beta score."""
    return tm_multilabel_fbeta_score(
        torch.utils.dlpack.from_dlpack(preds),
        torch.utils.dlpack.from_dlpack(target),
        beta=beta,
        num_labels=num_labels,
        threshold=threshold,
        average=average,
        ignore_index=ignore_index,
    )


@pytest.mark.parametrize("beta", [1.0, 2.0])
class TestMultilabelFBetaScore(MetricTester):
    """Test multilabel F-beta function and class."""

    atol = 2e-7

    @pytest.mark.parametrize("inputs", _multilabel_cases(xp=anp))
    @pytest.mark.parametrize("average", [None, "micro", "macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_multilabel_fbeta_score_with_numpy_array_api_arrays(
        self,
        inputs,
        beta,
        average,
        ignore_index,
    ) -> None:
        """Test function for multilabel fbeta score with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_function_implementation_test(
            target,
            preds,
            metric_function=multilabel_f1_score
            if beta == 1.0
            else partial(multilabel_fbeta_score, beta=beta),
            reference_metric=partial(
                _multilabel_fbeta_score_reference,
                beta=beta,
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
    def test_multilabel_fbeta_score_class_with_numpy_array_api_arrays(
        self,
        inputs,
        beta,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel fbeta score with `numpy.array_api` arrays."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelF1Score
            if beta == 1.0
            else partial(MultilabelFBetaScore, beta=beta),
            reference_metric=partial(
                _multilabel_fbeta_score_reference,
                beta=beta,
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
    def test_multilabel_fbeta_score_class_with_torch_tensors(
        self,
        inputs,
        beta,
        average,
        ignore_index,
    ) -> None:
        """Test class for multilabel fbeta score with torch tensors."""
        target, preds = inputs

        self.run_metric_class_implementation_test(
            target,
            preds,
            metric_class=MultilabelF1Score
            if beta == 1.0
            else partial(MultilabelFBetaScore, beta=beta),
            reference_metric=partial(
                _multilabel_fbeta_score_reference,
                beta=beta,
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


def test_top_k_multilabel_fbeta_score():
    """Test top-k multilabel fbeta score."""
    beta = 1.0
    target = anp.asarray([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = anp.asarray([[0.1, 0.9, 0.8, 0.3], [0.9, 0.1, 0.8, 0.3]])
    expected_result = anp.asarray([1.0, 1.0, 1.0, 0.0], dtype=anp.float32)

    result = multilabel_fbeta_score(
        target,
        preds,
        beta,
        num_labels=4,
        average=None,
        top_k=2,
    )
    assert np.allclose(result, expected_result)

    metric = MultilabelFBetaScore(beta, num_labels=4, average=None, top_k=2)
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
    expected_result = anp.asarray([0.4, 0.0, 0.0, 0.0], dtype=anp.float32)

    result = multilabel_fbeta_score(
        target,
        preds,
        beta,
        num_labels=4,
        average=None,
        top_k=2,
    )
    assert np.allclose(result, expected_result)

    class_result = metric(target, preds)
    assert np.allclose(class_result, expected_result)
