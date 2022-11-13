"""Functions for computing the receiver operating characteristic (ROC) curve."""

import warnings
from typing import List, Literal, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics._ranking import _binary_clf_curve

from cyclops.evaluate.metrics.functional.precision_recall_curve import (
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_update,
    _format_thresholds,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_update,
)
from cyclops.evaluate.metrics.utils import _check_thresholds


def _roc_compute_from_confmat(
    confmat: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC curve from a multi-threshold confusion matrix.

    Parameters
    ----------
        confmat : numpy.ndarray
            A multi-threshold confusion matrix of size (num_thresholds, 2, 2) or
            (num_thresholds, num_classes, 2, 2).
        thresholds : numpy.ndarray of floats, default=None

    Returns
    -------
        fpr : numpy.ndarray
            False positive rate.
        tpr : numpy.ndarray
            True positive rate.
        thresholds : numpy.ndarray
            Thresholds used to compute fpr and tpr.

    """
    tps = confmat[..., 1, 1]
    fns = confmat[..., 1, 0]
    fps = confmat[..., 0, 1]
    tns = confmat[..., 0, 0]

    tpr = np.divide(
        tps, tps + fns, out=np.zeros_like(tps, dtype=np.float64), where=(tps + fns) != 0
    )
    fpr = np.divide(
        fps, fps + tns, out=np.zeros_like(fps, dtype=np.float64), where=(fps + tns) != 0
    )

    # reverse order of arrays
    tpr = np.flip(tpr, axis=0)
    fpr = np.flip(fpr, axis=0)
    thresholds = np.flip(thresholds, axis=0)

    return fpr, tpr, thresholds


def _binary_roc_compute(
    state: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    thresholds: np.ndarray = None,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC curve for binary classification.

    Parameters
    ----------
        state : tuple of numpy.ndarray or numpy.ndarray
            If ``thresholds`` is not None, ``state`` is a multi-threshold confusion
            matrix. If ``thresholds`` is None, ``state`` is a tuple of (target, preds).
            probabilities.
        thresholds : numpy.ndarray, default=None
            Thresholds used to binarize the predicted probabilities. If None,
            the unique values of the predicted probabilities are used as
            thresholds.
        pos_label : int, optional
            The label of the positive class.

    Returns
    -------
        fpr : numpy.ndarray
            False positive rate.
        tpr : numpy.ndarray
            True positive rate.
        thresholds : numpy.ndarray
            Thresholds used to compute fpr and tpr.

    """
    if isinstance(state, np.ndarray) and thresholds is not None:
        fpr, tpr, thresholds = _roc_compute_from_confmat(state, thresholds)
    else:
        fps, tps, thresholds = _binary_clf_curve(
            y_true=state[0], y_score=state[1], pos_label=pos_label, sample_weight=None
        )

        # start the curve at (0, 0)
        fps = np.hstack((0, fps))
        tps = np.hstack((0, tps))
        thresholds = np.hstack((1, thresholds))

        if fps[-1] <= 0:
            warnings.warn(
                "No negative samples in `target`, false positive value should be"
                " meaningless. Returning zero array in false positive score",
                UserWarning,
            )
            fpr = np.zeros_like(thresholds)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            warnings.warn(
                "No positive samples in `target`, true positive value should be"
                " meaningless. Returning zero array in true positive score",
                UserWarning,
            )
            tpr = np.zeros_like(thresholds)
        else:
            tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def binary_roc_curve(
    target: ArrayLike,
    preds: ArrayLike,
    thresholds: Union[int, List[float], np.ndarray] = None,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC curve for binary classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If ``preds`` is not in
            the range [0, 1], a sigmoid function is applied to transform it to
            the range [0, 1].
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for computing the precision and recall scores.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        pos_label : int, optional
            The label of the positive class.

    Returns
    -------
        fpr : numpy.ndarray
            False positive rate.
        tpr : numpy.ndarray
            True positive rate.
        thresholds : numpy.ndarray
            Thresholds used to compute fpr and tpr.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import binary_roc_curve
        >>> target = [1, 0, 1, 0]
        >>> preds = [0.9, 0.2, 0.8, 0.3]
        >>> fpr, tpr, thresholds = binary_roc_curve(target, preds, thresholds=5)
        >>> fpr
        array([0. , 0. , 0. , 0.5, 1. ])
        >>> tpr
        array([0., 1., 1., 1., 1.])
        >>> thresholds
        array([1.  , 0.75, 0.5 , 0.25, 0.  ])

    """
    _check_thresholds(thresholds)

    target, preds = _binary_precision_recall_curve_format(target, preds)
    thresholds = _format_thresholds(thresholds)

    state = _binary_precision_recall_curve_update(target, preds, thresholds)

    return _binary_roc_compute(state, thresholds=thresholds, pos_label=pos_label)


def _multiclass_roc_compute(
    state: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    thresholds: np.ndarray = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the ROC curve for multiclass classification tasks.

    Parameters
    ----------
        state : numpy.ndarray or tuple of numpy.ndarray
            If ``thresholds`` is not None, ``state`` is a multi-threshold confusion
            matrix. If ``thresholds`` is None, ``state`` is a tuple of (target, preds).
        num_classes : int
            Number of classes.
        thresholds : numpy.ndarray, default=None
            Thresholds used for binarizing the predicted probabilities. If not
            None, must be a 1D numpy array of floats in the [0, 1] range and
            monotonically increasing.

    Returns
    -------
        fpr : numpy.ndarray or list of numpy.ndarray
            False positive rate. If ``threshold`` is not None, ``fpr`` is a 1d numpy
            array. Otherwise, ``fpr`` is a list of 1d numpy arrays, one for each
            class.
        tpr : numpy.ndarray or list of numpy.ndarray
            True positive rate. If ``threshold`` is not None, ``tpr`` is a 1d numpy
            array. Otherwise, ``tpr`` is a list of 1d numpy arrays, one for each class.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used to compute fpr and tpr. ``threshold`` is not None,
            thresholds is a 1d numpy array. Otherwise, thresholds is a list of
            1d numpy arrays, one for each class.

    """
    if isinstance(state, np.ndarray) and thresholds is not None:
        fpr, tpr, thresholds = _roc_compute_from_confmat(state, thresholds)

        tpr = tpr.T
        fpr = fpr.T
    else:
        fpr, tpr, thresholds = [], [], []
        for i in range(num_classes):
            res = _binary_roc_compute(
                [state[0], state[1][:, i]], thresholds=None, pos_label=i
            )
            fpr.append(res[0])
            tpr.append(res[1])
            thresholds.append(res[2])

    return fpr, tpr, thresholds


def multiclass_roc_curve(
    target: ArrayLike,
    preds: ArrayLike,
    num_classes: int,
    thresholds: Union[int, List[float], np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the ROC curve for multiclass classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If ``preds`` is not in
            the range [0, 1], a softmax function is applied to transform it to
            the range [0, 1].
        num_classes : int
            Number of classes.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for binarizing the predicted probabilities.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Returns
    -------
        fpr : numpy.ndarray or list of numpy.ndarray
            False positive rate. If ``threshold`` is not None, ``fpr`` is a 1d numpy
            array. Otherwise, ``fpr`` is a list of 1d numpy arrays, one for each
            class.
        tpr : numpy.ndarray or list of numpy.ndarray
            True positive rate. If ``threshold`` is not None, ``tpr`` is a 1d numpy
            array. Otherwise, ``tpr`` is a list of 1d numpy arrays, one for each class.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used to compute fpr and tpr. ``threshold`` is not None,
            thresholds is a 1d numpy array. Otherwise, thresholds is a list of
            1d numpy arrays, one for each class.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multiclass_roc_curve
        >>> target = [1, 0, 2, 0]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05],
        ...         [0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]
        >>> fpr, tpr, thresholds = multiclass_roc_curve(target, preds,
        ...     num_classes=3, thresholds=5
        ... )
        >>> fpr
        array([[0.        , 0.5       , 0.5       , 0.5       , 1.        ],
        [0.        , 0.33333333, 0.33333333, 0.33333333, 1.        ],
        [0.        , 0.        , 0.        , 0.        , 1.        ]])
        >>> tpr
        array([[0. , 0.5, 0.5, 0.5, 1. ],
        [0. , 0. , 0. , 0. , 1. ],
        [0. , 1. , 1. , 1. , 1. ]])
        >>> thresholds
        array([1.  , 0.75, 0.5 , 0.25, 0.  ])

    """
    _check_thresholds(thresholds)
    target, preds = _multiclass_precision_recall_curve_format(
        target, preds, num_classes=num_classes
    )
    thresholds = _format_thresholds(thresholds)

    state = _multiclass_precision_recall_curve_update(
        target, preds, num_classes=num_classes, thresholds=thresholds
    )

    return _multiclass_roc_compute(state, num_classes, thresholds)


def _multilabel_roc_compute(
    state: Union[Tuple[np.ndarray, np.ndarray], np.ndarray],
    num_labels: int,
    thresholds: np.ndarray = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the ROC curve for multilabel classification tasks.

    Parameters
    ----------
        state : numpy.ndarray or tuple of numpy.ndarray
            If ``thresholds`` is not None, ``state`` is a multi-threshold confusion
            matrix. Otherwise, ``state`` is a tuple of (target, preds).
        num_labels : int
            Number of labels.
        thresholds : numpy.ndarray, default=None
            Thresholds used for binarizing the predicted probabilities. If not
            None, must be a 1D numpy array of floats in the [0, 1] range and
            monotonically increasing.

    Returns
    -------
        fpr : numpy.ndarray or list of numpy.ndarray
            False positive rate. If ``threshold`` is not None, ``fpr`` is a 1d numpy
            array. Otherwise, ``fpr`` is a list of 1d numpy arrays, one for each
            label.
        tpr : numpy.ndarray or list of numpy.ndarray
            True positive rate. If ``threshold`` is not None, ``tpr`` is a 1d numpy
            array. Otherwise, ``tpr`` is a list of 1d numpy arrays, one for each label.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used to compute fpr and tpr. ``threshold`` is not None,
            thresholds is a 1d numpy array. Otherwise, thresholds is a list of
            1d numpy arrays, one for each label.

    """
    if isinstance(state, np.ndarray) and thresholds is not None:
        fpr, tpr, thresholds = _roc_compute_from_confmat(state, thresholds)

        tpr = tpr.T
        fpr = fpr.T
    else:
        fpr, tpr, thresholds = [], [], []
        for i in range(num_labels):
            res = _binary_roc_compute(
                [state[0][:, i], state[1][:, i]], thresholds=None, pos_label=i
            )
            fpr.append(res[0])
            tpr.append(res[1])
            thresholds.append(res[2])

    return fpr, tpr, thresholds


def multilabel_roc_curve(
    target: ArrayLike,
    preds: ArrayLike,
    num_labels: int,
    thresholds: Union[int, List[float], np.ndarray] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the ROC curve for multilabel classification tasks.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or decision function. If ``preds`` is not in
            the range [0, 1], a sigmoid function is applied to transform it to
            the range [0, 1].
        num_labels : int
            The number of labels in the dataset.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Returns
    -------
        fpr : numpy.ndarray or list of numpy.ndarray
            False positive rate. If ``threshold`` is not None, ``fpr`` is a 1d numpy
            array. Otherwise, ``fpr`` is a list of 1d numpy arrays, one for each
            label.
        tpr : numpy.ndarray or list of numpy.ndarray
            True positive rate. If ``threshold`` is not None, ``tpr`` is a 1d numpy
            array. Otherwise, ``tpr`` is a list of 1d numpy arrays, one for each label.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used to compute fpr and tpr. ``threshold`` is not None,
            thresholds is a 1d numpy array. Otherwise, thresholds is a list of
            1d numpy arrays, one for each label.

    Examples
    --------
        >>> from cyclops.evaluation.metrics.functional import multilabel_roc_curve
        >>> target = [[0, 1, 0], [0, 1, 1], [1, 0, 1]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.1, 0.9], [0.8, 0.2, 0.3]]
        >>> fpr, tpr, thresholds = multilabel_roc_curve(target, preds, num_labels=3,
        ...     thresholds=5
        ... )
        >>> fpr
        array([[0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1.],
        [0., 1., 1., 1., 1.]])
        >>> tpr
        array([[0. , 1. , 1. , 1. , 1. ],
        [0. , 0.5, 0.5, 0.5, 1. ],
        [0. , 0.5, 0.5, 1. , 1. ]])
        >>> thresholds
        array([1.  , 0.75, 0.5 , 0.25, 0.  ])

    """
    _check_thresholds(thresholds)
    target, preds = _multilabel_precision_recall_curve_format(
        target, preds, num_labels=num_labels
    )
    thresholds = _format_thresholds(thresholds)

    state = _multilabel_precision_recall_curve_update(
        target, preds, num_labels=num_labels, thresholds=thresholds
    )

    return _multilabel_roc_compute(state, num_labels, thresholds)


def roc_curve(  # pylint: disable=too-many-arguments
    target: ArrayLike,
    preds: ArrayLike,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Union[int, List[float], np.ndarray] = None,
    pos_label: int = 1,
    num_classes: int = None,
    num_labels: int = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
]:
    """Compute the ROC curve for different tasks/input types.

    Parameters
    ----------
        target : ArrayLike
            Ground truth (correct) target values.
        preds : ArrayLike
            Estimated probabilities or non-thresholded output of decision function.
            If ``task`` is ``multiclass`` and the values in ``preds`` are not
            probabilities, they will be converted to probabilities using the softmax
            function. If ``task`` is ``multilabel`` and the values in ``preds`` are
            not probabilities, they will be converted to probabilities using the
            sigmoid function.
        task : Literal["binary", "multiclass", "multilabel"]
            The type of task for the input data. One of 'binary', 'multiclass'
            or 'multilabel'.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for computing the ROC curve. Can be one of:
                - None: use the unique values of ``preds`` as thresholds
                - int: generate ``thresholds`` number of evenly spaced values between
                  0 and 1 as thresholds.
                - list of floats: use the values in the list as thresholds. The list
                  of values should be monotonically increasing. The list will be
                  converted into a numpy array.
                - numpy.ndarray of floats: use the values in the array as thresholds.
                  The array should be 1d and monotonically increasing.
        pos_label : int, default=1
            The label of the positive class.
        num_classes : int, optional
            The number of classes in the dataset. Required for multiclass tasks.
        num_labels : int, optional
            The number of labels in the dataset. Required for multilabel tasks.

    Returns
    -------
        fpr : numpy.ndarray or list of numpy.ndarray
            False positive rate. If ``task`` is 'binary' or ``threshold`` is not None,
            ``fpr`` is a 1d numpy array. If ``task`` is 'multiclass' or 'multilabel',
            and ``threshold`` is None, then ``fpr`` is a list of 1d numpy
            arrays, one for each class or label.
        tpr : numpy.ndarray or list of numpy.ndarray
            True positive rate. If ``task`` is 'binary' or ``threshold`` is not None,
            ``tpr`` is a 1d numpy array. If ``task`` is 'multiclass' or 'multilabel',
            and ``threshold`` is None, then ``tpr`` is a list of 1d numpy
            arrays, one for each class or label.
        thresholds : numpy.ndarray or list of numpy.ndarray
            Thresholds used to compute fpr and tpr. If ``task`` is 'binary' or
            ``threshold`` is not None, ``thresholds`` is a 1d numpy array. If
            ``task`` is 'multiclass' or 'multilabel', and ``threshold`` is None,
            then ``thresholds`` is a list of 1d numpy arrays, one for each class
            or label.

    Raises
    ------
        ValueError
            If ``task`` is not one of 'binary', 'multiclass' or 'multilabel'.
        AssertionError
            If ``task`` is ``multiclass`` and ``num_classes`` is not provided.
        AssertionError
            If ``task`` is ``multilabel`` and ``num_labels`` is not provided.

    Example (binary)
    ----------------
        >>> from cyclops.evaluation.metrics.functional import roc_curve
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> fpr, tpr, thresholds = roc_curve(target, preds, task='binary')
        >>> fpr
        array([0. , 0. , 0.5, 0.5, 1. ])
        >>> tpr
        array([0. , 0.5, 0.5, 1. , 1. ])
        >>> thresholds
        array([1.  , 0.8 , 0.4 , 0.35, 0.1 ])

    Example (multiclass)
    --------------------
        >>> from cyclops.evaluation.metrics.functional import roc_curve
        >>> target = [0, 1, 2]
        >>> preds = [[0.9, 0.05, 0.05], [0.05, 0.89, 0.06], [0.02, 0.03, 0.95]]
        >>> fpr, tpr, thresholds = roc_curve(target, preds, task='multiclass',
        ...     num_classes=3
        ... )
        >>> fpr
        [array([0. , 0. , 0.5, 1. ]),
        array([0. , 0. , 0.5, 1. ]),
        array([0. , 0. , 0.5, 1. ])]
        >>> tpr
        [array([0., 1., 1., 1.]), array([0., 1., 1., 1.]), array([0., 1., 1., 1.])]
        >>> thresholds
        [array([1.  , 0.9 , 0.05, 0.02]),
        array([1.  , 0.89, 0.05, 0.03]),
        array([1.  , 0.95, 0.06, 0.05])]

    Example (multilabel)
    --------------------
        >>> from cyclops.evaluation.metrics.functional import roc_curve
        >>> target = [[1, 1], [0, 1], [1, 0]]
        >>> preds = [[0.9, 0.8], [0.2, 0.7], [0.8, 0.3]]
        >>> fpr, tpr, thresholds = roc_curve(target, preds, task='multilabel',
        ...     num_labels=2
        ... )
        >>> fpr
        [array([0. , 0.5, 1. , 1. ]), array([0., 0., 0., 1.])]
        >>> tpr
        [array([0., 0., 0., 1.]), array([0. , 0.5, 1. , 1. ])]
        >>> thresholds
        [array([1. , 0.9, 0.8, 0.2]), array([1. , 0.8, 0.7, 0.3])]

    """
    _check_thresholds(thresholds)
    if task == "binary":
        fpr, tpr, thresholds = binary_roc_curve(
            target, preds, thresholds, pos_label=pos_label
        )
    elif task == "multiclass":
        assert isinstance(
            num_classes, int
        ), "Number of classes must be a positive integer."
        fpr, tpr, thresholds = multiclass_roc_curve(
            target, preds, num_classes, thresholds
        )
    elif task == "multilabel":
        assert isinstance(
            num_labels, int
        ), "Number of labels must be a positive integer."
        fpr, tpr, thresholds = multilabel_roc_curve(
            target, preds, num_labels, thresholds
        )
    else:
        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}"
        )

    return fpr, tpr, thresholds
