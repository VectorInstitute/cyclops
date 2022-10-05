from typing import Optional, Tuple

import numpy as np
from sklearn import preprocessing
from numpy.typing import ArrayLike
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

# Boolean, unsigned integer, signed integer, float, complex.
_NUMERIC_KINDS = set("buifc")


def is_numeric(*arrays: ArrayLike) -> bool:
    """Determine whether the argument(s) have a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Arguments
    ---------
    arrays: array-likes
        The arrays to check.

    Returns
    -------
    is_numeric: `bool`
        True if all of the arrays have a numeric datatype, False if not.
    """
    return all(np.asanyarray(array).dtype.kind in _NUMERIC_KINDS for array in arrays)


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def _input_squeeze(
    target: np.ndarray, preds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove excess dimensions."""
    if preds.shape[0] == 1:
        target, preds = np.expand_dims(target.squeeze(), axis=0), np.expand_dims(
            preds.squeeze(), axis=0
        )
    else:
        target, preds = target.squeeze(), preds.squeeze()
    return target, preds


def _check_muldim_input(target: np.ndarray, preds: np.ndarray) -> None:
    """Check if the input is multidimensional. None of the metrics support
    multidimensional input.

    Arguments
    ----------
        preds: np.ndarray
            The predictions.
        target: np.ndarray
            The target.

    Raises
    ------
        ValueError
            If the input is multidimensional.
    """
    if preds.ndim > 2 or target.ndim > 2:
        raise ValueError(
            "preds and target should be 1D or 2D arrays. "
            f"Got {preds.ndim}D and {target.ndim}D arrays."
        )


def _common_input_checks_and_format(
    target: ArrayLike,
    preds: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Check the input and convert it to the correct format. This function
    also checks if the input is valid.

    Arguments
    ----------
        target: ArrayLike
            The target.
        preds: ArrayLike
            The predictions.

    Returns
    -------
        target: np.ndarray
            The target as a numpy array.
        preds: np.ndarray
            The predictions as a numpy array.
        type_target: str
            The type of the target. One of:

        * 'continuous': `target` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `target` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `target` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `target` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `target` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `target` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `target` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

        type_preds: str
            The type of the predictions.

    Raises
    ------
        ValueError
            If the input has more than two dimensions.
    """
    target, preds = np.asanyarray(target), np.asanyarray(preds)

    _check_muldim_input(
        target, preds
    )  # multidimensional-multiclass input is not supported

    target, preds = _input_squeeze(target, preds)

    type_target = type_of_target(target)
    type_preds = type_of_target(preds)

    return target, preds, type_target, type_preds


def _str_to_categorical(*inputs, labels=None):
    """Converts string labels to categorical labels."""
    ret = []
    for input in inputs:
        if input is not None:
            input = np.asanyarray(input)
            if input.dtype.kind == "U" and labels is None:
                input = preprocessing.LabelEncoder().fit_transform(input)
            elif input.dtype.kind == "U" and labels is not None:
                input = preprocessing.LabelEncoder().fit(labels).transform(input)
        ret.append(input)

    return tuple(ret)


def _check_topk(top_k: int, type_preds: str, type_target: str, n_classes: int) -> None:
    """Check if top_k is valid.

    Arguments
    ----------
        top_k: int
            The number of classes to select.
        n_classes: int
            The number of classes.

    Raises
    ------
        ValueError
            If top_k is not valid.
    """
    if type_target == "binary":
        raise ValueError("You can not use `top_k` parameter with binary data.")
    if type_preds not in ["continuous", "continuous-multioutput"]:
        raise ValueError("You can only use top_k with continuous predictions.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("The `top_k` has to be an integer larger than 0.")
    if top_k >= n_classes:
        raise ValueError(
            "The `top_k` has to be strictly smaller than the number of classes."
        )


def _select_topk(prob_scores: np.ndarray, top_k: Optional[int] = 1) -> np.ndarray:
    """Convert a probability scores to binary by selecting top-k highest entries.

    Arguments
    ----------
        prob_scores: np.ndarray
            The probability scores. Must be a 2D array.
        top_k: int
            The number of top predictions to select. Defaults to 1.

    Returns
    -------
        A binary ndarray of the same shape as the input ndarray of type np.int32
    """
    if top_k == 1:
        topk_indices = np.argmax(prob_scores, axis=1, keepdims=True)
    else:
        topk_indices = np.argsort(prob_scores, axis=1)[:, ::-1][
            :, :top_k
        ]  # sort in descending order, then slice the top k

    topk_array = np.zeros_like(prob_scores)
    np.put_along_axis(topk_array, topk_indices, 1.0, axis=1)

    return topk_array.astype(np.int32)


def _input_classification_format(
    target: ArrayLike,
    preds: ArrayLike,
    threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
    labels: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert continuous prediction inputs to the format required by the
    classification metrics.

    Arguments
    ----------
        target: ArrayLike
            The target.
        preds: ArrayLike
            The predictions.
        threshold: float
            Threshold for logits and probability targets. Defaults to 0.5.
        top_k: int
            The number of top predictions to select. Defaults to 1.
        labels: ArrayLike
            The set of labels to include.

    Returns
    -------
        A tuple of the following elements in order: target, preds.
    """
    target, preds = _common_input_checks_and_format(target, preds)

    n_classes = preds.shape[1] if len(preds.shape) == 2 else 2
    if labels is not None:
        n_classes = len(labels)

    if top_k is not None:
        _check_topk(top_k, type_preds, type_target, n_classes)

    if type_preds in ["continuous", "continuous-multioutput"]:
        if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
            preds = sigmoid(preds)  # convert logits to probabilities

        if target.dtype.type is np.str_ and type_target != "multilabel-indicator":
            target = preprocessing.LabelEncoder().fit_transform(target)

    # binary case
    if type_preds == "continuous" or (
        type_preds == "continuous-multioutput"
        and type_target == "multilabel-indicator"
        and not top_k
    ):
        preds = (preds > threshold).astype(np.int32)

    if type_preds == "continuous-multioutput":
        if type_target == "multilabel-indicator":
            if top_k is not None:
                preds = _select_topk(preds, top_k)

        if type_target == "multiclass":
            preds = _select_topk(preds, top_k or 1)
            target = np.eye(preds.shape[1])[target]  # one-hot encoding

    return target, preds


def _stat_scores_update(
    target: ArrayLike,
    preds: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    reduce: Optional[str] = "micro",
    threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
):
    """Update and return variables required to compute stat scores.

    Arguments
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        sample_weight: ArrayLike
            Sample weights.
        labels: ArrayLike
            The set of labels to include.
        reduce: String
            The method to reduce the stat scores over labels. One of:

            * 'micro': Calculate statistics globally by counting the total true positives,
                false negatives and false positives.
            * 'macro': Calculate statistics for each label.
            * 'samples': Calculate statistics for each instance. Only available for
                multilabel targets.

            Defaults to ``'micro'``.
        threshold: float
            Threshold for converting logits and probability predictions to binary [1, 0].
            Defaults to 0.5.
        top_k: int
            The number of top predictions to select. Defaults to 1.

    Returns
    -------
        A tuple of the following elements in order: true positives, false positives,
        false negatives, true negatives, and the support (number of samples).
    """

    if not is_numeric(target, preds):
        raise ValueError("The target and preds must be numeric.")

    target, preds = _input_classification_format(
        target, preds, threshold, top_k, labels
    )

    # convert labels to indices if target is already converted
    if labels is not None and target.dtype.type is not np.str_:
        labels = np.asanyarray(labels)
        labels = preprocessing.LabelEncoder().fit_transform(labels)

    tp, fp, tn, fn = _stat_scores(
        target, preds, sample_weight=sample_weight, labels=labels, reduce=reduce
    )

    return tp, fp, tn, fn


def _stat_scores(
    target: np.ndarray,
    preds: np.ndarray,
    sample_weight: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    reduce: Optional[str] = "micro",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the number of true positives, false positives, true negatives

    Arguments
    ----------
        preds: np.ndarray
            Predictions.
        target: np.ndarray
            Ground truth.
        sample_weight: np.ndarray
            Sample weights.
        labels: np.ndarray
            The set of labels
        reduce: String
            Reduction mode. Defaults to 'micro'.

    Returns
    -------
        The stat scores.
    """
    samplewise = reduce == "samples"
    multi_confusion = multilabel_confusion_matrix(
        target, preds, sample_weight=sample_weight, labels=labels, samplewise=samplewise
    )  # shape: (num_outputs, 2, 2)

    tn = multi_confusion[:, 0, 0]  # shape: (num_outputs,)
    fn = multi_confusion[:, 1, 0]
    tp = multi_confusion[:, 1, 1]
    fp = multi_confusion[:, 0, 1]

    if reduce == "micro":
        tp = tp.sum(keepdims=True)
        fp = fp.sum(keepdims=True)
        tn = tn.sum(keepdims=True)
        fn = fn.sum(keepdims=True)

    return (
        tp.astype(np.int32),
        fp.astype(np.int32),
        tn.astype(np.int32),
        fn.astype(np.int32),
    )


def _stat_scores_compute(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Computes the number of true positives, false positives, true negatives
    and false negatives. Concatenates the results in a single array, along with
    the support.

    Arguments
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

    output: np.ndarray = np.concatenate(stats, axis=-1).squeeze()
    ouput = np.where(output < 0, -1, output)

    return output


def stat_scores(
    target: ArrayLike,
    preds: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    reduce: Optional[str] = "micro",
    threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Computes the number of true positives, false positives, true negatives
    and false negatives. Concatenates the results in a single array, along with
    the support.

    Arguments
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        sample_weight: ArrayLike
            Sample weights.
        labels: ArrayLike
            The set of labels to include.
        reduce: String
            The method to reduce the stat scores over labels. One of:

            * 'micro': Calculate statistics globally by counting the total true positives,
                false negatives and false positives.
            * 'macro': Calculate statistics for each label.
            * 'samples': Calculate statistics for each instance. Only available for
                multilabel targets.

            Defaults to ``'micro'``.
        threshold: float
            Threshold for converting logits and probability predictions to binary [1, 0].
            Defaults to 0.5.
        top_k: int
            Number of the highest probability entries for each sample to convert
            to 1s - relevant only for inputs with probability predictions.
            The default value (``None``) will be interpreted as 1 for multiclass inputs.
            If this parameter is set for multi-label inputs, it will take precedence
            over threshold.
            Should be left unset (``None``) for inputs with label predictions.

    Returns
    -------
        The stat scores.
    """
    if reduce not in ["micro", "macro", "samples"]:
        raise ValueError(
            "The argument `reduce` must be one of 'micro', 'macro', 'samples'"
        )

    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(
            f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}."
        )

    tp, fp, tn, fn = _stat_scores_update(
        target,
        preds,
        sample_weight=sample_weight,
        labels=labels,
        reduce=reduce,
        threshold=threshold,
        top_k=top_k,
    )
    return _stat_scores_compute(tp, fp, tn, fn)


def _binary_stat_scores_format(target: ArrayLike, preds: ArrayLike, threshold: float):
    """Formats the binary stat scores. Converts the target and preds to binary
    and checks the shape.

    Arguments
    ---------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        threshold: float
            Threshold for converting logits and probability predictions to binary [1, 0].

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
    target, preds, type_target, type_preds = _common_input_checks_and_format(
        target, preds
    )

    if type_target != "binary" or type_preds not in ["binary", "continuous"]:
        raise ValueError(
            "The arguments `target` and `preds` must be binary or continuous."
        )

    # check the number of classes
    unique_values = np.unique(target)
    check = any((unique_values != 0) & (unique_values != 1))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0,1]}."
        )

    # If preds is label array, also check that it only contains [0,1] values
    if not type_preds == "continuous":
        unique_values = np.unique(preds)
        if any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since `preds` is a label array."
            )

    if type_preds == "continuous":
        if not np.all(np.logical_and(preds >= 0.0, preds <= 1.0)):
            preds = sigmoid(preds)  # convert logits to probabilities

        preds = (preds > threshold).astype(np.int32)

    return target, preds


def _binary_stat_scores_update(
    target: ArrayLike,
    preds: ArrayLike,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    normalize: Optional[str] = None,
    threshold: Optional[float] = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the true positives, false positives, true negatives and false
    negatives for binary inputs.

    Arguments
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        labels: ArrayLike
            The set of labels to include.
        sample_weight: ArrayLike
            Sample weights.
        normalize: String
            The method to normalize the stat scores. One of:

            * 'true': Divide the stat scores by the number of true positives.
            * 'pred': Divide the stat scores by the number of predicted positives.
            * 'all': Divide the stat scores by the total number of samples.

            Defaults to ``None``.
        threshold: float
            Threshold for converting logits and probability predictions to binary [1, 0].
            Defaults to 0.5.

    Returns
    -------
        The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If the target and preds are not numeric.

    """
    if not is_numeric(target, preds):
        raise ValueError("The input `target` and `preds` must be numeric.")

    target, preds = _binary_stat_scores_format(target, preds, threshold)

    cm = confusion_matrix(
        target, preds, labels=labels, sample_weight=sample_weight, normalize=normalize
    )

    tn, fp, fn, tp = cm.ravel()

    return (
        tp.astype(np.int32),
        fp.astype(np.int32),
        tn.astype(np.int32),
        fn.astype(np.int32),
    )


def _binary_stat_scores_compute(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    """Returns the stat scores for binary inputs.

    Arguments
    ---------
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
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    normalize: Optional[str] = None,
    threshold: Optional[float] = 0.5,
) -> np.ndarray:
    """Computes the true positives, false positives, true negatives and false
    negatives for binary inputs.

    Arguments
    ----------
        target: ArrayLike
            Ground truth.
        preds: ArrayLike
            Predictions.
        labels: ArrayLike
            The set of labels to include.
        sample_weight: ArrayLike
            Sample weights.
        normalize: String
            The method to normalize the stat scores. One of:

            * 'true': Divide the stat scores by the number of true positives.
            * 'pred': Divide the stat scores by the number of predicted positives.
            * 'all': Divide the stat scores by the total number of samples.

            Defaults to ``None``.
        threshold: float
            Threshold for converting logits and probability predictions to binary [1, 0].
            Defaults to 0.5.

    Returns
    -------
        The true positives, false positives, true negatives and false negatives.

    Raises
    ------
        ValueError
            If threshold is not a float and not in [0,1].


    """
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(
            f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}."
        )

    tp, fp, tn, fn = _binary_stat_scores_update(
        target,
        preds,
        labels=labels,
        sample_weight=sample_weight,
        normalize=normalize,
        threshold=threshold,
    )

    return _binary_stat_scores_compute(tp, fp, tn, fn)


def _multiclass_stat_scores_format(
    target: ArrayLike, preds: ArrayLike, num_classes: int, top_k: int = 1
):
    target, preds, type_target, type_preds = _common_input_checks_and_format(
        target, preds
    )
    # determine if the input can be coerced into multiclass format
    if not (type_target in ["binary", "multiclass"]):
        raise ValueError(
            f"The argument `target` must be binary or multiclass, got {type_target}."
        )

    num_implied_classes = len(np.unique(target))
    if num_implied_classes > num_classes:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes} but found {num_implied_classes} in `target`."
        )

    if type_preds == "binary" and num_classes > 2:
        type_preds = "multiclass"
    if not type_preds in ["multiclass", "continuous-multioutput"]:
        raise ValueError(
            f"The argument `preds` must be multiclass or continuous multioutput, got {type_preds}."
        )

    if type_preds != "continuous-multioutput":
        num_implied_classes = len(np.unique(preds))
        if num_implied_classes > num_classes:
            raise RuntimeError(
                "Detected more unique values in `preds` than `num_classes`. Expected only "
                f"{num_classes} but found {num_implied_classes} in `preds`."
            )

    labels = np.arange(num_classes)

    if top_k is not None:
        _check_topk(top_k, type_preds, type_target, num_classes)

    if type_preds == "continuous-multioutput" and not np.all(
        np.logical_and(preds >= 0.0, preds <= 1.0)
    ):
        preds = sigmoid(preds)  # convert logits to probabilities

    if type_preds == "continuous-multioutput" and type_target == "multiclass":
        preds = _select_topk(preds, top_k or 1)
        target = np.eye(preds.shape[1])[target]  # one-hot encoding

    return target, preds, labels


def _multiclass_stat_scores_update(
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    sample_weight: Optional[ArrayLike] = None,
    classwise: Optional[bool] = False,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not is_numeric(target, preds):
        raise ValueError("The input `target` and `preds` must be numeric.")

    target, preds, labels = _multiclass_stat_scores_format(
        target, preds, num_classes, top_k
    )

    reduce = "macro" if classwise else "micro"
    tp, fp, tn, fn = _stat_scores(
        target, preds, sample_weight=sample_weight, labels=labels, reduce=reduce
    )

    return tp, fp, tn, fn


def _multiclass_stat_scores_compute(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    return _stat_scores_compute(tp, fp, tn, fn)


def multiclass_stat_scores(
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    sample_weight: Optional[ArrayLike] = None,
    classwise: Optional[bool] = False,
    top_k: Optional[int] = None,
) -> np.ndarray:
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
):
    target, preds, type_target, type_preds = _common_input_checks_and_format(
        target, preds
    )

    if not type_target == "multilabel-indicator":
        raise ValueError(
            f"The argument `target` must be multilabel-indicator, got {type_target}."
        )

    if not (type_preds in ["multilabel-indicator", "continuous-multioutput"]):
        raise ValueError(
            f"The argument `preds` must be multilabel-indicator, or continuous multioutput, got {type_preds}."
        )

    implied_num_labels = preds.shape[1]
    if implied_num_labels != num_labels:
        raise RuntimeError(
            f"Detected {implied_num_labels} labels in `preds` but expected {num_labels}."
        )

    if top_k is not None:
        _check_topk(top_k, type_preds, type_target, num_labels)

    labels = np.arange(num_labels)

    if type_preds == "continuous-multioutput" and not np.all(
        np.logical_and(preds >= 0.0, preds <= 1.0)
    ):
        preds = sigmoid(preds)  # convert logits to probabilities

    if type_preds == "continuous-multioutput":
        if top_k is not None:
            preds = _select_topk(preds, top_k)
        else:
            preds = (preds >= threshold).astype(np.int32)

    return target, preds, labels


def _multilabel_stat_scores_update(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    sample_weight: Optional[ArrayLike] = None,
    reduce: Optional[str] = "micro",
    threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not is_numeric(target, preds):
        raise ValueError("The input `target` and `preds` must be numeric.")

    target, preds, labels = _multilabel_stat_scores_format(
        target, preds, num_labels, threshold, top_k
    )

    tp, fp, tn, fn = _stat_scores(
        target, preds, sample_weight=sample_weight, labels=labels, reduce=reduce
    )

    return tp, fp, tn, fn


def _multilabel_stat_scores_compute(
    tp: np.ndarray, fp: np.ndarray, tn: np.ndarray, fn: np.ndarray
) -> np.ndarray:
    return _stat_scores_compute(tp, fp, tn, fn)


def multilabel_stat_scores(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    sample_weight: Optional[ArrayLike] = None,
    reduce: Optional[str] = "micro",
    threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
) -> np.ndarray:
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
