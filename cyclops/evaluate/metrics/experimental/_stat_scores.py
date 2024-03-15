"""Abstract classes for computing true/false positive/negative scores."""

from typing import Any, Literal, Optional, Tuple, Union

from cyclops.evaluate.metrics.experimental.confusion_matrix import (
    _AbstractConfusionMatrix,
)
from cyclops.evaluate.metrics.experimental.functional._stat_scores import (
    _binary_stat_scores_format_arrays,
    _binary_stat_scores_update_state,
    _binary_stat_scores_validate_args,
    _binary_stat_scores_validate_arrays,
    _multiclass_stat_scores_format_arrays,
    _multiclass_stat_scores_update_state,
    _multiclass_stat_scores_validate_args,
    _multiclass_stat_scores_validate_arrays,
    _multilabel_stat_scores_format_arrays,
    _multilabel_stat_scores_update_state,
    _multilabel_stat_scores_validate_args,
    _multilabel_stat_scores_validate_arrays,
)
from cyclops.evaluate.metrics.experimental.utils.types import Array


class _AbstractBinaryStatScores(_AbstractConfusionMatrix):
    """The number of true/false positives/negatives for a binary classifier."""

    name: str = "Stat Scores"

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        _binary_stat_scores_validate_args(
            threshold=threshold,
            ignore_index=ignore_index,
        )

        self.threshold = threshold
        self.ignore_index = ignore_index

        self._create_state(size=1)

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the statistic scores with the given inputs.

        Parameters
        ----------
        target : Array
            An array object that is compatible with the Python array API standard
            and contains the ground truth labels. The expected shape of the array
            is `(N, ...)`, where `N` is the number of samples.
        preds : Array
            An array object that is compatible with the Python array API standard and
            contains the predictions of a binary classifier. The expected shape of the
            array is `(N, ...)` where `N` is the number of samples. If `preds` contains
            floating point values that are not in the range `[0, 1]`, a sigmoid function
            will be applied to each value before thresholding.

        """
        xp = _binary_stat_scores_validate_arrays(
            target,
            preds,
            ignore_index=self.ignore_index,
        )
        target, preds = _binary_stat_scores_format_arrays(
            target,
            preds,
            threshold=self.threshold,
            ignore_index=self.ignore_index,
            xp=xp,
        )
        tn, fp, fn, tp = _binary_stat_scores_update_state(target, preds, xp=xp)
        self._update_stat_scores(tp=tp, fp=fp, tn=tn, fn=fn)


class _AbstractMulticlassStatScores(_AbstractConfusionMatrix):
    """The number of true/false positives/negatives for a multiclass classifier."""

    name: str = "Stat Scores"

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        ignore_index: Optional[Union[int, Tuple[int]]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        _multiclass_stat_scores_validate_args(
            num_classes,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
        )
        self.num_classes = num_classes
        self.top_k = top_k
        self.average = average
        self.ignore_index = ignore_index

        self._create_state(
            size=1 if (average == "micro" and top_k == 1) else num_classes,
        )

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the statistic scores with the given inputs.

        Parameters
        ----------
        target : Array
            An array object that is compatible with the Python array API standard
            and contains the ground truth labels. The expected shape of the array
            is `(N, ...)`, where `N` is the number of samples.
        preds : Array
            An array object that is compatible with the Python array API standard and
            contains the predictions of a classifier. If `preds` contains integer values
            the expected shape of the array is `(N, ...)`, where `N` is the number of
            samples. If `preds` contains floating point values the expected shape of the
            array is `(N, C, ...)` where `N` is the number of samples and `C` is the
            number of classes.
        """
        xp = _multiclass_stat_scores_validate_arrays(
            target,
            preds,
            self.num_classes,
            top_k=self.top_k,
            ignore_index=self.ignore_index,
        )
        target, preds = _multiclass_stat_scores_format_arrays(
            target,
            preds,
            top_k=self.top_k,
            xp=xp,
        )
        tn, fp, fn, tp = _multiclass_stat_scores_update_state(
            target,
            preds,
            self.num_classes,
            top_k=self.top_k,
            average=self.average,
            ignore_index=self.ignore_index,
            xp=xp,
        )
        self._update_stat_scores(tp=tp, fp=fp, tn=tn, fn=fn)


class _AbstractMultilabelStatScores(_AbstractConfusionMatrix):
    """The number of true/false positives/negatives for a multilabel classifier."""

    name: str = "Stat Scores"

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        top_k: int = 1,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        _multilabel_stat_scores_validate_args(
            num_labels,
            threshold=threshold,
            top_k=top_k,
            average=average,
            ignore_index=ignore_index,
        )
        self.num_labels = num_labels
        self.threshold = threshold
        self.top_k = top_k
        self.average = average
        self.ignore_index = ignore_index

        self._create_state(size=num_labels)

    def _update_state(self, target: Array, preds: Array) -> None:
        """Update the statistic scores with the given inputs.

        Parameters
        ----------
        target : Array
            An array object that is compatible with the Python array API standard
            and contains the ground truth labels. The expected shape of the array
            is `(N, L, ...)`, where `N` is the number of samples and `L` is the
            number of labels.
        preds : Array
            An array object that is compatible with the Python array API standard and
            contains the predictions of a classifier. The expected shape of the array
            is `(N, L, ...)`, where `N` is the number of samples and `L` is the
            number of labels. If `preds` contains floating point values that are not
            in the range `[0, 1]`, a sigmoid function will be applied to each value
            before thresholding.
        """
        xp = _multilabel_stat_scores_validate_arrays(
            target,
            preds,
            self.num_labels,
            ignore_index=self.ignore_index,
        )
        target, preds = _multilabel_stat_scores_format_arrays(
            target,
            preds,
            threshold=self.threshold,
            top_k=self.top_k,
            ignore_index=self.ignore_index,
            xp=xp,
        )
        tn, fp, fn, tp = _multilabel_stat_scores_update_state(target, preds, xp=xp)
        self._update_stat_scores(tp=tp, fp=fp, tn=tn, fn=fn)
