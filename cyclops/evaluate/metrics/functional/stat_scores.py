"""Functions for computing stat scores for different types of inputs.

The stat scores are the number of true positives, false positives, true negatives, and
false negatives. Binary, multiclass and multilabel data are supported, including logits
and probabilities.

"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize

from cyclops.evaluate.metrics.utils import (
    check_topk,
    common_input_checks_and_format,
    select_topk,
    sigmoid,
)


def _stat_scores_compute(  # pylint: disable=invalid-name
    tp: Union[np.ndarray, np.int_],
    fp: Union[np.ndarray, np.int_],
    tn: Union[np.ndarray, np.int_],
    fn: Union[np.ndarray, np.int_],
) -> np.ndarray:
    """Compute true positives, false positives, true negatives and false negatives.

    Concatenates the results in a single array, along with the support.

    Parameters
    ----------
        tp : numpy.ndarray or numpy.int_
            True positives.
        fp : numpy.ndarray or numpy.int_
            False positives.
        tn : numpy.ndarray or numpy.int_
            True negatives.
        fn : numpy.ndarray or numpy.int_
            False negatives.

    Returns
    -------
        The stat scores.

    """
    if tp.ndim == 1 and tp.size == 1:  # 1D array with 1 element
        stats = [tp, fp, tn, fn, tp + fn]
    else:
        stats = [
            np.expand_dims(tp, axis=-1),
            np.expand_dims(fp, axis=-1),
            np.expand_dims(tn, axis=-1),
            np.expand_dims(fn, axis=-1),
            np.expand_dims(tp, axis=-1) + np.expand_dims(fn, axis=-1),  # support
        ]

    output: np.ndarray = np.concatenate(stats, axis=-1)

    return output


def _stat_scores_from_confmat(
    target: np.ndarray,
    preds: np.ndarray,
    labels: Optional[ArrayLike] = None,
    classwise: Optional[bool] = False,
) -> Tuple[
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
]:
    """Compute true positives, false positives, true negatives and false negatives.

    Parameters
    ----------
        preds : numpy.ndarray
            Predictions.
        target : numpy.ndarray
            Ground truth.
        labels : numpy.ndarray, default=None
            The set of labels to include.
        classwise : bool, default=True
            If True, compute the stat scores for each class separately. Otherwise,
            compute the stat scores for the whole array.

    Returns
    -------
        Tuple of true positives, false positives, true negatives and false negatives.

    """
    # pylint: disable=invalid-name
    confmat = multilabel_confusion_matrix(
        target, preds, labels=labels
    )  # shape: (n_classes, 2, 2)

    tn = confmat[:, 0, 0]  # shape: (n_classes,)
    fn = confmat[:, 1, 0]
    tp = confmat[:, 1, 1]
    fp = confmat[:, 0, 1]

    if not classwise:
        tp = tp.sum()
        fp = fp.sum()
        tn = tn.sum()
        fn = fn.sum()

    return (
        tp.astype(np.int_),
        fp.astype(np.int_),
        tn.astype(np.int_),
        fn.astype(np.int_),
    )


def _binary_stat_scores_args_check(threshold: float, pos_label: int) -> None:
    """Check the arguments for binary stat scores.

    Parameters
    ----------
        threshold : float
            Threshold for converting logits and probability predictions to binary
            [1, 0].
        pos_label : int
            The positive label to report.

    Returns
    -------
        None

    Raises
    ------
        ValueError
            If the threshold is not in [0, 1] or if the pos_label is not 0 or 1.

    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")

    if pos_label not in [0, 1]:
        raise ValueError(f"Positive label must be 0 or 1, got {pos_label}.")


def _binary_stat_scores_format(
    target: ArrayLike, preds: ArrayLike, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Format the input for computing binary stat scores.

    Checks that ``target`` and ``preds`` are binary and have the same shape.
    If ``preds`` is in continuous form, it is binarized using the given threshold.
    Logits are converted to probabilities using the sigmoid function.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        threshold : float
            Threshold for converting logits and probability predictions to binary
            [1, 0].

    Returns
    -------
        Tuple[numpy.ndarray, numpy.ndarray]
            The formatted target and preds as numpy.ndarray.

    Raises
    ------
        ValueError
            If the target and preds are not binary.

        ValueError
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
        raise ValueError(
            f"Detected the following values in `target`: {unique_values} but"
            f" expected only the following values {[0,1]}."
        )

    # If preds is label array, also check that it only contains [0,1] values
    if not type_preds == "continuous":
        unique_values = np.unique(preds)
        if any((unique_values != 0) & (unique_values != 1)):
            raise ValueError(
                f"Detected the following values in `preds`: {unique_values} but"
                f" expected only [0,1] values since `preds` is a label array."
            )

    if type_preds == "continuous":
        if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
            preds = sigmoid(preds)  # convert logits to probabilities

        preds = (preds >= threshold).astype(np.int_)

    return target.astype(np.int_), preds.astype(np.int_)


def _binary_stat_scores_update(
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
) -> Tuple[
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
]:
    """Compute the statistics for binary inputs.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        pos_label : int, default=1
            The positive label to report. Can be either 0, 1.

    Returns
    -------
        Tuple[Union[numpy.ndarray, numpy.int_], Union[numpy.ndarray, numpy.int_],
        Union[numpy.ndarray, numpy.int_], Union[numpy.ndarray, numpy.int_]]
            The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If the target and preds are not numeric.

    """
    return _stat_scores_from_confmat(target, preds, labels=[pos_label], classwise=True)


def binary_stat_scores(
    target: ArrayLike,
    preds: ArrayLike,
    pos_label: int = 1,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute the stat scores for binary inputs.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        pos_label : int, default=1
            The label to use for the positive class.
        threshold : float, default=0.5
            The threshold to use for converting the predictions to binary
            values. Logits will be converted to probabilities using the sigmoid
            function.

    Returns
    -------
        numpy.ndarray
            The true positives, false positives, true negatives and false negatives
            and support in that order.

    Raises
    ------
        ValueError
            If the threshold is not in [0, 1] or if the pos_label is not 0 or 1.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_stat_scores
        >>> target = [0, 1, 1, 0]
        >>> preds = [0, 1, 0, 0]
        >>> binary_stat_scores(target, preds)
        array([1, 0, 2, 1, 2])

    """
    _binary_stat_scores_args_check(threshold=threshold, pos_label=pos_label)

    target, preds = _binary_stat_scores_format(
        target=target, preds=preds, threshold=threshold
    )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _binary_stat_scores_update(
        target=target, preds=preds, pos_label=pos_label
    )

    return _stat_scores_compute(tp=tp, fp=fp, tn=tn, fn=fn)


def _multiclass_stat_scores_format(
    target: ArrayLike, preds: ArrayLike, num_classes: int, top_k: Optional[int] = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Format the target and preds for multiclass inputs.

    Checks that the target and preds are of the same length and that the target
    and preds are of the correct shape. Converts the target and preds to the
    correct type.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        num_classes : int
            The total number of classes for the problem.
        top_k : int
            The number of top predictions to consider when computing the statistics.
            Defaults to 1.

    Returns
    -------
        Tuple[numpy.ndarray, numpy.ndarray]
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
            f"The argument `target` must be multiclass, got {type_target}."
        )

    num_implied_classes = len(np.unique(target))
    if num_implied_classes > num_classes:
        raise ValueError(
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
            raise ValueError(
                "Detected more unique values in `preds` than `num_classes`. Expected "
                f"only {num_classes} but found {num_implied_classes} in `preds`."
            )

    # check top_k
    if top_k is not None:
        check_topk(top_k, type_preds, type_target, num_classes)

    # handle probabilities and logits
    if type_preds == "continuous-multioutput":
        if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
            preds = sp.special.softmax(preds, axis=1)  # convert logits to probabilities

        if not np.allclose(1, preds.sum(axis=1)):
            raise ValueError(
                "``preds`` need to be probabilities for multiclass problems"
                " i.e. they should sum up to 1.0 over classes"
            )

        # convert ``preds`` and ``target`` to multilabel-indicator format
        preds = select_topk(preds, top_k or 1)
        target = label_binarize(target, classes=np.arange(num_classes))

    return target.astype(np.int_), preds.astype(np.int_)


def _multiclass_stat_scores_update(  # pylint: disable=too-many-arguments
    target: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
    classwise: Optional[bool] = True,
) -> Tuple[
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
]:
    """Update the stat scores for multiclass inputs.

    Parameters
    ----------
        target : numpy.ndarray
            Ground truth.
        preds : numpy.ndarray
            Predictions.
        num_classes : int
            The total number of classes for the problem.
        classwise : bool, default=True
            Whether to return the statistics for each class or sum over all classes.

    Returns
    -------
        Tuple[Union[numpy.ndarray, numpy.int_], Union[numpy.ndarray, numpy.int_],
        Union[numpy.ndarray, numpy.int_], Union[numpy.ndarray, numpy.int_]]
            The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If the input target and preds are not numeric.

    """
    return _stat_scores_from_confmat(
        target, preds, labels=np.arange(num_classes), classwise=classwise
    )


def multiclass_stat_scores(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    top_k: Optional[int] = None,
    classwise: Optional[bool] = True,
) -> np.ndarray:
    """Compute stat scores for multiclass targets.

    Parameters
    ----------
        target : ArrayLike
            The ground truth values.
        preds : ArrayLike
            The predictions. If determined to be in continuous format, will be
            converted to multiclass using the ``top_k`` parameter.
        num_classes : int
            The total number of classes for the problem.
        top_k : Optional[int], default=None
            The number of top predictions to consider when computing the
            stat scores. If ``None``, it is assumed to be 1.
        classwise : bool, default=True
            Whether to return the stat scores for each class or sum over all
            classes.

    Returns
    -------
        numpy.nadarray
            The number of true positives, false positives, true negatives, false
            negatives and support. If ``classwise`` is ``True``, the shape is
            ``(num_classes, 5)``. Otherwise, the shape is ``(5,)``

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_stat_scores
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 2, 1, 2, 0]
        >>> multiclass_stat_scores(target, preds, num_classes=3)
        array([[1, 1, 3, 0, 1],
               [0, 1, 3, 1, 1],
               [1, 1, 1, 2, 3]])

    """
    target, preds = _multiclass_stat_scores_format(
        target=target, preds=preds, num_classes=num_classes, top_k=top_k
    )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        target=target,
        preds=preds,
        num_classes=num_classes,
        classwise=classwise,
    )

    return _stat_scores_compute(tp=tp, fp=fp, tn=tn, fn=fn)


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
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        num_labels : int
            The total number of labels for the problem.
        threshold : float, default=0.5
            Threshold value for binarizing the predictions.
        top_k : int, default=None
            The number of top predictions to consider when computing the statistics.

    Returns
    -------
        Tuple[numpy.ndarray, numpy.ndarray]
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
        raise ValueError(
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
            preds = (preds >= threshold).astype(np.int_)

    return target.astype(np.int_), preds.astype(np.int_)


def _multilabel_stat_scores_update(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    labelwise: Optional[bool] = False,
) -> Tuple[
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
    Union[np.ndarray, np.int_],
]:
    """Update the stat scores for multilabel inputs.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        num_labels : int
            The total number of labels for the problem.
         labelwise : bool, default=False
            Whether to return the statistics for each label or sum over all labels.

    Returns
    -------
        numpy.ndarray
            The number of true positives, false positives, true negatives and false
            negatives.

    Raises
    ------
        ValueError
            If the input target and preds are not numeric.

    """
    return _stat_scores_from_confmat(
        target, preds, labels=np.arange(num_labels), classwise=labelwise
    )


def multilabel_stat_scores(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    labelwise: Optional[bool] = False,
) -> np.ndarray:
    """Compute the stat scores for multilabel inputs.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        num_labels : int
            The total number of labels for the problem.
        threshold : float, default=0.5
            Threshold value for binarizing predictions that are probabilities or
            logits. A sigmoid function is applied if the predictions are logits.
        top_k : int, default=None
            The number of top predictions to consider when computing the statistics.
        labelwise : bool, default=False
            Whether to return the stat scores for each label or sum over all labels.

    Returns
    -------
        numpy.ndarray
            The number of true positives, false positives, true negatives and false
            negatives and the support. The shape of the array is ``(5, num_labels)``
            if ``labelwise=True`` and ``(5,)`` otherwise.

    Raises
    ------
        ValueError
            If ``threshold`` is not between ``0`` and ``1``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_stat_scores
        >>> target = [[0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]]
        >>> multilabel_stat_scores(target, preds, num_labels=3)
        array([[1, 0, 1, 0, 1],
               [1, 0, 1, 0, 1],
               [2, 0, 0, 0, 2]])

    """
    _binary_stat_scores_args_check(threshold=threshold, pos_label=1)

    target, preds = _multilabel_stat_scores_format(
        target=target,
        preds=preds,
        num_labels=num_labels,
        threshold=threshold,
        top_k=top_k,
    )

    # pylint: disable=invalid-name
    tp, fp, tn, fn = _multilabel_stat_scores_update(
        target=target,
        preds=preds,
        num_labels=num_labels,
        labelwise=labelwise,
    )

    return _stat_scores_compute(tp, fp, tn, fn)


def stat_scores(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    pos_label: int = 1,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    classwise: Optional[bool] = True,
    top_k: Optional[int] = None,
    num_labels: Optional[int] = None,
    labelwise: Optional[bool] = False,
) -> np.ndarray:
    """Compute stat scores for binary, multiclass or multilabel problems.

    This function acts as an entry point to the specialized functions for each
    task.

    Parameters
    ----------
        target : ArrayLike
            Ground truth.
        preds : ArrayLike
            Predictions.
        task : Literal["binary", "multiclass", "multilabel"]
            The task type. Can be either ``binary``, ``multiclass`` or
            ``multilabel``.
        pos_label : int, default=1
            The positive label to report. Only used for binary tasks.
        threshold : float, default=0.5
            The threshold to use for binarizing the predictions if logits or
            probabilities are provided. If logits are provided, a sigmoid function
            is applied prior to binarization. Used for binary and multilabel tasks.
        num_classes : int
            The number of classes for the problem. Required for multiclass tasks.
        classwise : bool, default=True
            Whether to return the stat scores for each class or sum over all
            classes. Only used for multiclass tasks.
        top_k : int, default=None
            The number of top predictions to consider when computing the statistics.
            If ``None``, ``top_k`` is set to 1. Used for multiclass and multilabel
            tasks.
        num_labels : int
            The number of labels. Only used for multilabel tasks.
        labelwise : bool, default=False
            Whether to compute the stat scores labelwise. Only used for multilabel
            tasks.

    Returns
    -------
        scores : numpy.ndarray
            The stat scores - true positives, false positives, true negatives,
            false negatives and support. For binary tasks, the shape is (5,).
            For multiclass tasks, the shape is (n_classes, 5) if ``classwise`` is
            True, otherwise (5,). For multilabel tasks, the shape is (n_labels, 5)
            if ``labelwise`` is True, otherwise (n_classes, 5).

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics.functional import tat_scores
        >>> target = [0, 1, 1, 0]
        >>> preds = [0, 1, 0, 0]
        >>> stat_scores(target, preds, task="binary")
        array([1, 0, 2, 1, 2])

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import multiclass_stat_scores
        >>> target = [0, 1, 2, 2, 2]
        >>> preds = [0, 2, 1, 2, 0]
        >>> stat_scores(target, preds, task="multiclass", num_classes=3)
        array([[1, 1, 3, 0, 1],
               [0, 1, 3, 1, 1],
               [1, 1, 1, 2, 3]])

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics.functional import stat_scores
        >>> target = [[0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.8, 0.2, 0.7]]
        >>> stat_scores(target, preds, task="multilabel", num_labels=3)
        array([[1, 0, 1, 0, 1],
               [1, 0, 1, 0, 1],
               [2, 0, 0, 0, 2]])

    """
    if task == "binary":
        scores = binary_stat_scores(
            target,
            preds,
            pos_label=pos_label,
            threshold=threshold,
        )
    elif task == "multiclass":
        assert (
            isinstance(num_classes, int) and num_classes > 0
        ), "Number of classes must be a positive integer."
        scores = multiclass_stat_scores(
            target,
            preds,
            num_classes,
            classwise=classwise,
            top_k=top_k,
        )
    elif task == "multilabel":
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "Number of labels must be a positive integer."
        scores = multilabel_stat_scores(
            target,
            preds,
            num_labels,
            labelwise=labelwise,
            threshold=threshold,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"Unsupported task: {task}, expected one of 'binary', 'multiclass' or "
            f"'multilabel'."
        )

    return scores
