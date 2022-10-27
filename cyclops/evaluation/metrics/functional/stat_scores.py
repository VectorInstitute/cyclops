"""Functions for computing stat scores for different types of inputs.

The stat scores are the number of true positives, false positives, true negatives, and
false negatives. Binary, multiclass and multilabel data are supported, including logits
and probabilities.

"""
from typing import Literal, Optional, Tuple

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize

from cyclops.evaluation.metrics.utils import (
    check_topk,
    common_input_checks_and_format,
    select_topk,
    sigmoid,
)


def _stat_scores_update(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    pos_label: int = 1,
    num_classes: Optional[int] = None,
    classwise: Optional[bool] = False,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    num_labels: Optional[int] = None,
    sample_weight: Optional[ArrayLike] = None,
    reduce: Optional[Literal["micro", "macro", "samples"]] = "micro",
):
    """Update the stat scores for a given task.

    Parameters
    ----------
        preds: ArrayLike
            Predictions.
        target: ArrayLike
            Ground truth.
        task: String
            The task type. Can be either 'binary', 'multiclass' or 'multilabel'.
        pos_label: int
            The positive label to report. Defaults to 1. Can be either 0, 1.
            Only applicable to binary data.
        num_classes: int
            The number of classes. Only used for multiclass tasks.
        num_labels: int
            The number of labels. Only used for multilabel tasks.
        sample_weight: ArrayLike
            Sample weights.
        classwise: bool
            Whether to compute the stat scores classwise. Defaults to False. Only
            used for multiclass tasks.
        reduce: String
            Reduction mode. Defaults to 'macro'. One of:
            'micro': sum the statistics over all samples and labels to compute the
                global score.
            'macro': Calculate metrics for each label/class.
            'samples': Calculate metrics for each instance/sample.
        threshold: float
            Threshold for binarizing the predictions. Defaults to 0.5. Only used
            for binary and multilabel tasks.
        top_k: int
            Number of top elements to look at for computing accuracy. Only used
            for multiclass and multilabel tasks.

    Returns
    -------
        The stat scores.

    Raises
    ------
        ValueError
            If the task is not one of 'binary', 'multiclass' or 'multilabel'.

    """
    # pylint: disable=invalid-name
    if task == "binary":
        tp, fp, tn, fn = _binary_stat_scores_update(
            target,
            preds,
            pos_label=pos_label,
            threshold=threshold,
            sample_weight=sample_weight,
        )
    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be a positive integer."
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            target,
            preds,
            num_classes,
            sample_weight=sample_weight,
            classwise=classwise,
            top_k=top_k,
        )
    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be a positive integer."
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            target,
            preds,
            num_labels,
            sample_weight=sample_weight,
            reduce=reduce,
            threshold=threshold,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"Unsupported task: {task}, expected one of 'binary', 'multiclass' or "
            f"'multilabel'."
        )

    return tp, fp, tn, fn


def _stat_scores(
    target: np.ndarray,
    preds: np.ndarray,
    sample_weight: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    reduce: Optional[Literal["micro", "macro", "samples"]] = "micro",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute true positives, false positives, true negatives and false negatives.

    Parameters
    ----------
        preds: np.ndarray
            Predictions.
        target: np.ndarray
            Ground truth.
        sample_weight: np.ndarray
            Sample weights.
        labels: np.ndarray
            The set of labels to include.
        reduce: String
            Reduction mode. Defaults to 'micro'.

    Returns
    -------
        The stat scores.

    """
    # pylint: disable=invalid-name
    samplewise = reduce == "samples"
    confmat = multilabel_confusion_matrix(
        target, preds, sample_weight=sample_weight, labels=labels, samplewise=samplewise
    )  # shape: (num_outputs, 2, 2)

    tn = confmat[:, 0, 0]  # shape: (num_outputs,)
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    fp = confmat[:, 0, 1]

    if reduce == "micro":
        tp = tp.sum()
        fp = fp.sum()
        tn = tn.sum()
        fn = fn.sum()

    return (
        tp.astype(np.int32),
        fp.astype(np.int32),
        tn.astype(np.int32),
        fn.astype(np.int32),
    )


def _stat_scores_compute(  # pylint: disable=invalid-name
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Compute true positives, false positives, true negatives and false negatives.

    Concatenates the results in a single array, along with the support.

    Parameters
    ----------
        tp: np.ndarray
            True positives.
        fp: np.ndarray
            False positives.
        tn: np.ndarray
            True negatives.
        fn: np.ndarray
            False negatives.

    Returns
    -------
        The stat scores.

    """
    stats = [
        np.expand_dims(tp, axis=-1),
        np.expand_dims(fp, axis=-1),
        np.expand_dims(tn, axis=-1),
        np.expand_dims(fn, axis=-1),
        np.expand_dims(tp, axis=-1) + np.expand_dims(fn, axis=-1),  # support
    ]

    output: np.ndarray = np.concatenate(stats, axis=-1)
    output = np.where(output < 0, -1, output)

    return output


def stat_scores(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    pos_label: int = 1,
    num_classes: Optional[int] = None,
    classwise: Optional[bool] = False,
    top_k: Optional[int] = None,
    threshold: float = 0.5,
    num_labels: Optional[int] = None,
    reduce: Optional[Literal["micro", "macro", "samples"]] = "micro",
    sample_weight: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute true positives, false positives, true negatives and false negatives.

    Concatenates the results in a single array, along with the support. This
    function acts as an entry point for more specialized functions.

    Parameters
    ----------
        preds: ArrayLike
            Predictions.
        target: ArrayLike
            Ground truth.
        task: String
            The task type. Can be either ``binary``, ``multiclass`` or
            ``multilabel``. One of:
                - ``binary``: binary classification.
                    Example: [0, 1, 1, 0, 1] or [0.1, 0.9, 0.8, 0.2, 0.4]
                - ``multiclass``: multiclass classification.
                    Example: [0, 1, 2, 0, 1] or [[0.1, 0.9, 0.0], [0.0, 0.8, 0.2], ...]
                - ``multilabel``: multilabel classification.
                    Example: [[0, 1], [1, 0], [1, 1], [0, 0], [1, 0]] or
                    [[0.1, 0.9], [0.0, 0.8], ...]

        pos_label: int
            The positive label. Defaults to 1. Only used for binary tasks.
        num_classes: int
            The number of classes. Only used for multiclass tasks.
        classwise: bool
            Whether to compute the stat scores classwise. Only used for multiclass
            tasks. Defaults to False.
        top_k: int
            The number of top classes to consider. Used for multiclass and
            multilabel tasks. Defaults to None.
        threshold: float
            The threshold to use for binarizing the predictions. Used for
            binary and multilabel tasks. Defaults to 0.5.
        num_labels: int
            The number of labels. Only used for multilabel tasks.
        reduce: Literal["micro", "macro", "samples"]
            Reduction mode. Defaults to 'micro'. Only used for multiclass and
            multilabel tasks. One of:
                - ``micro``: compute the stat scores globally.
                - ``macro``: compute the stat scores for each class.
                - ``samples``: compute the stat scores for each sample.

        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The stat scores.

    """
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _stat_scores_update(
        target,
        preds,
        task,
        pos_label=pos_label,
        num_classes=num_classes,
        num_labels=num_labels,
        sample_weight=sample_weight,
        classwise=classwise,
        reduce=reduce,
        threshold=threshold,
        top_k=top_k,
    )

    return _stat_scores_compute(tp, fp, tn, fn)


def _binary_stat_scores_format(target: ArrayLike, preds: ArrayLike, threshold: float):
    """Format the input for computing binary stat scores.

    Converts the target and preds to binary and checks the shape.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        threshold: float
            Threshold for converting logits and probability predictions to binary
            [1, 0].

    Returns
    -------
        The formatted target and preds.

    Raises
    ------
        ValueError
            If the target and preds are not binary.

        RuntimeError
            If the target and preds have non-binary values.

    """
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target, preds
    )

    if type_target != "binary" or type_preds not in ["binary", "continuous"]:
        raise ValueError(
            "The arguments `target` and `preds` must be binary or continuous, "
            f"got {type_target} and {type_preds} respectively."
        )

    # check the number of classes
    unique_values = np.unique(target)
    check = any((unique_values != 0) & (unique_values != 1))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but"
            f" expected only the following values {[0,1]}."
        )

    # If preds is label array, also check that it only contains [0,1] values
    if not type_preds == "continuous":
        unique_values = np.unique(preds)
        if any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but"
                f" expected only [0,1] values since `preds` is a label array."
            )

    if type_preds == "continuous":
        if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
            preds = sigmoid(preds)  # convert logits to probabilities

        preds = (preds >= threshold).astype(np.int32)

    return target.astype(np.int32), preds.astype(np.int32)


def _binary_stat_scores_update(
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
    sample_weight: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute the statistics for binary inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        pos_label: int
            The positive label to report. Defaults to 1. Can be either 0, 1.
        threshold: float
            Threshold for converting logits and probability predictions to binary
            [1, 0]. Defaults to 0.5.
        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If the target and preds are not numeric.

    """
    # pylint: disable=invalid-name
    target, preds = _binary_stat_scores_format(target, preds, threshold)

    if pos_label not in [0, 1]:
        raise ValueError(
            f"The argument `pos_label` must be either 0 or 1, got {pos_label}."
        )

    confmat = multilabel_confusion_matrix(
        target, preds, sample_weight=sample_weight, labels=[pos_label]
    )

    tn = confmat[:, 0, 0]
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    fp = confmat[:, 0, 1]

    return np.concatenate([tp, fp, tn, fn]).astype(np.int32)


def _binary_stat_scores_compute(  # pylint: disable=invalid-name
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Return the stat scores for binary inputs.

    Parameters
    ----------
        tp: np.ndarray
            True positives.
        fp: np.ndarray
            False positives.
        tn: np.ndarray
            True negatives.
        fn: np.ndarray
            False negatives.

    Returns
    -------
        The stat scores.

    """
    return _stat_scores_compute(tp, fp, tn, fn)


def binary_stat_scores(
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: Optional[float] = 0.5,
    sample_weight: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute the stat scores for binary inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        pos_label: int
            The positive label to report. Defaults to 1. Can be either 0, 1.
        threshold: float
            Threshold for converting logits and probability predictions to binary
            [1, 0]. Defaults to 0.5.
        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If threshold is not a float and not in [0,1].

    """
    # pylint: disable=invalid-name
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(
            f"Expected argument `threshold` to be a float in the [0,1] range, "
            f"but got {threshold}."
        )

    tp, fp, tn, fn = _binary_stat_scores_update(
        target,
        preds,
        pos_label=pos_label,
        sample_weight=sample_weight,
        threshold=threshold,
    )

    return _binary_stat_scores_compute(tp, fp, tn, fn)


def _multiclass_stat_scores_format(
    target: ArrayLike, preds: ArrayLike, num_classes: int, top_k: Optional[int] = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Format the target and preds for multiclass inputs.

    Checks that the target and preds are of the same length and that the target
    and preds are of the correct shape. Converts the target and preds to the
    correct type.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        num_classes: int
            The total number of classes for the problem.
        top_k: int
            The number of top predictions to consider when computing the statistics.
            Defaults to 1.

    Returns
    -------
        The formatted target and preds.

    Raises
    ------
        ValueError
            If the target is not in binary (with maximum label > 1) or multiclass
            format.
        RuntimeError
            If more unique values are detected in `target` than `num_classes`.
        ValueError
            If the predictions are not in multiclass or continuous-multioutput
            (logits or probabilites) format.
        RuntimeError
            If more unique values are detected in `preds` than `num_classes`.

    """
    # convert target and preds to numpy arrays
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target, preds
    )

    # check the target
    if type_target not in ["binary", "multiclass"]:
        raise ValueError(
            f"The argument `target` must be binary or multiclass, got {type_target}."
        )

    num_implied_classes = len(np.unique(target))
    if num_implied_classes > num_classes:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes} but found {num_implied_classes} in `target`."
        )

    # check the preds
    if type_preds == "binary" and num_classes > 2:
        type_preds = "multiclass"
    if type_preds not in ["multiclass", "continuous-multioutput"]:
        raise ValueError(
            f"The argument `preds` must be multiclass or continuous multioutput, "
            f"got {type_preds}."
        )

    if type_preds != "continuous-multioutput":
        num_implied_classes = len(np.unique(preds))
        if num_implied_classes > num_classes:
            raise RuntimeError(
                "Detected more unique values in `preds` than `num_classes`. Expected "
                f"only {num_classes} but found {num_implied_classes} in `preds`."
            )

    # check top_k
    if top_k is not None:
        check_topk(top_k, type_preds, type_target, num_classes)

    # handle probabilities and logits
    if type_preds == "continuous-multioutput" and not np.all(
        np.logical_and(preds >= 0.0, preds <= 1.0)
    ):
        preds = sp.special.softmax(preds, axis=1)  # convert logits to probabilities

    if type_preds == "continuous-multioutput" and type_target == "multiclass":
        preds = select_topk(preds, top_k or 1)
        target = label_binarize(target, classes=np.arange(num_classes))

    return target.astype(np.int32), preds.astype(np.int32)


def _multiclass_stat_scores_update(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    classwise: Optional[bool] = False,
    sample_weight: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update the stat scores for multiclass inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        num_classes: int
            The total number of classes for the problem.
        top_k: int
            The number of top predictions to consider when computing the statistics.
            Defaults to ``None``.
        classwise: bool
            Whether to return the statistics for each class or sum over all classes.
            Defaults to ``False``.
        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If the input target and preds are not numeric.

    """
    # pylint: disable=invalid-name
    target, preds = _multiclass_stat_scores_format(target, preds, num_classes, top_k)
    labels = np.arange(num_classes)

    reduce: Literal["micro", "macro", "samples"] = "macro" if classwise else "micro"
    tp, fp, tn, fn = _stat_scores(
        target, preds, sample_weight=sample_weight, labels=labels, reduce=reduce
    )

    return tp, fp, tn, fn


def _multiclass_stat_scores_compute(  # pylint: disable=invalid-name
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Compute the stat scores for multiclass inputs.

    Parameters
    ----------
        tp: np.ndarray
            True positives.
        fp: np.ndarray
            False positives.
        tn: np.ndarray
            True negatives.
        fn: np.ndarray
            False negatives.

    Returns
    -------
        The true positives, false positives, true negatives and false negatives.

    """
    return _stat_scores_compute(tp, fp, tn, fn)


def multiclass_stat_scores(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    classwise: Optional[bool] = False,
    sample_weight: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute stat scores for multiclass inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        num_classes: int
            The total number of classes for the problem.
        top_k: int
            The number of top predictions to consider when computing the statistics.
            Defaults to ``None``.
        classwise: bool
            Whether to return the statistics for each class or sum over all classes.
            Defaults to ``False``.
        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The number of true positives, false positives, true negatives and false
        negatives.

    """
    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        target,
        preds,
        sample_weight=sample_weight,
        num_classes=num_classes,
        classwise=classwise,
        top_k=top_k,
    )

    return _multiclass_stat_scores_compute(tp, fp, tn, fn)


def _multilabel_stat_scores_format(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Format the target and preds for multilabel inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        num_labels: int
            The total number of labels for the problem.
        threshold: float
            Threshold value for binarizing the predictions. Defaults to ``0.5``.
        top_k: int
            The number of top predictions to consider when computing the statistics.

    Returns
    -------
        The formatted target and preds.

    Raises
    ------
        ValueError
            If the target is not in multilabel format.
        ValueError
            If the predictions are not in multilabel or continuous-multioutput
            (probabilities or logits) format.
        RuntimeError
            If the number of labels implied by the predictions is inconsistent with
            ``num_labels``.

    """
    target, preds, type_target, type_preds = common_input_checks_and_format(
        target, preds
    )

    if not type_target == "multilabel-indicator":
        raise ValueError(
            f"The argument `target` must be multilabel-indicator, got {type_target}."
        )

    if not (type_preds in ["multilabel-indicator", "continuous-multioutput"]):
        raise ValueError(
            f"The argument `preds` must be multilabel-indicator, or continuous "
            f"multioutput, got {type_preds}."
        )

    implied_num_labels = preds.shape[1]
    if implied_num_labels != num_labels:
        raise RuntimeError(
            f"Detected {implied_num_labels} labels in `preds` but expected "
            f"{num_labels}."
        )

    if top_k is not None:
        check_topk(top_k, type_preds, type_target, num_labels)

    if type_preds == "continuous-multioutput" and not np.all(
        np.logical_and(preds >= 0.0, preds <= 1.0)
    ):
        preds = sigmoid(preds)

    if type_preds == "continuous-multioutput":
        if top_k is not None:
            preds = select_topk(preds, top_k)
        else:
            preds = (preds >= threshold).astype(np.int32)

    return target.astype(np.int32), preds.astype(np.int32)


def _multilabel_stat_scores_update(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    reduce: Optional[Literal["micro", "macro", "samples"]] = "micro",
    sample_weight: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update the stat scores for multilabel inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        num_labels: int
            The total number of labels for the problem.
        threshold: float
            Threshold value for binarizing the predictions. Defaults to ``0.5``.
        top_k: int
            The number of top predictions to consider when computing the statistics.
            Defaults to ``None``.
         reduce: str
            The reduction method to use. Defaults to ``"micro"``. Can be one of
                * ``"micro"`` - sum the statistics over all labels.
                * ``"macro"`` - compute the unweighted mean of the per-label
                statistics.
                * ``"samples"`` - compute the unweighted mean of the per-sample
                statistics.
        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The number of true positives, false positives, true negatives and false
        negatives.

    Raises
    ------
        ValueError
            If the input target and preds are not numeric.

    """
    # pylint: disable=invalid-name
    target, preds = _multilabel_stat_scores_format(
        target, preds, num_labels, threshold, top_k
    )
    labels = np.arange(num_labels)

    tp, fp, tn, fn = _stat_scores(
        target, preds, sample_weight=sample_weight, labels=labels, reduce=reduce
    )

    return tp, fp, tn, fn


def _multilabel_stat_scores_compute(  # pylint: disable=invalid-name
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Compute the stat scores for multilabel inputs.

    Parameters
    ----------
        tp: np.ndarray
            The number of true positives.
        fp: np.ndarray
            The number of false positives.
        tn: np.ndarray
            The number of true negatives.
        fn: np.ndarray
            The number of false negatives.

    Returns
    -------
        The number of true positives, false positives, true negatives and false
        negatives.

    """
    return _stat_scores_compute(tp, fp, tn, fn)


def multilabel_stat_scores(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
    reduce: Optional[Literal["micro", "macro", "samples"]] = "micro",
    sample_weight: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute the stat scores for multilabel inputs.

    Parameters
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        num_labels: int
            The total number of labels for the problem.
        threshold: float
            Threshold value for binarizing the predictions. Defaults to ``0.5``.
        top_k: int
            The number of top predictions to consider when computing the statistics.
            Defaults to ``None``.
        reduce: Literal["micro", "macro", "samples"]
            The reduction method to use. Defaults to ``micro``. Can be one of:
                - ``micro`` - sum the statistics over all labels.
                - ``macro`` - compute the unweighted mean of the per-label statistics.
                - ``samples`` - compute the unweighted mean of the per-sample
                  statistics.
        sample_weight: ArrayLike
            Sample weights.

    Returns
    -------
        The number of true positives, false positives, true negatives and false
        negatives and the support.

    Raises
    ------
        ValueError
            If ``reduce`` is not one of ``micro``, ``macro`` or ``samples``.
        ValueError
            If ``threshold`` is not between ``0`` and ``1``.

    """
    if reduce not in ["micro", "macro", "samples"]:
        raise ValueError(
            "The argument `reduce` must be one of 'micro', 'macro', 'samples'"
        )

    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(
            f"Expected argument `threshold` to be a float in the [0,1] range, "
            f"but got {threshold}."
        )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multilabel_stat_scores_update(
        target,
        preds,
        num_labels,
        sample_weight=sample_weight,
        reduce=reduce,
        threshold=threshold,
        top_k=top_k,
    )

    return _multilabel_stat_scores_compute(tp, fp, tn, fn)
