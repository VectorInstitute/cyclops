"""Classes for computing ROC metrics."""

from typing import List, Literal, Tuple, Union

import numpy as np

from cyclops.evaluate.metrics import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.functional.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from cyclops.evaluate.metrics.metric import Metric

# mypy: ignore-errors


class BinaryROCCurve(BinaryPrecisionRecallCurve):
    """Compute the ROC curve for binary classification tasks.

    Parameters
    ----------
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for computing the precision and recall scores.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        pos_label : int, default=1
            The label of the positive class.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryROCCurve
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> metric = BinaryROCCurve()
        >>> metric(target, preds)
        (array([0. , 0. , 0.5, 0.5, 1. ]),
        array([0. , 0.5, 0.5, 1. , 1. ]),
        array([1.  , 0.8 , 0.4 , 0.35, 0.1 ]))
        >>> metric.reset_state()
        >>> target = [[1, 1, 0, 0], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.2, 0.3, 0.4], [0.6, 0.5, 0.4, 0.3]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.  , 1.  ]),
        array([0.  , 0.  , 0.  , 0.25, 0.5 , 0.75, 1.  ]),
        array([1. , 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))

    """

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the ROC curve from the state variables."""
        # pylint: disable=no-member # attributes are set with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _binary_roc_compute(
            state, thresholds=self.thresholds, pos_label=self.pos_label
        )


class MulticlassROCCurve(MulticlassPrecisionRecallCurve):
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

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassROCCurve
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.05, 0.95, 0], [0.1, 0.8, 0.1],
        ...         [0.2, 0.2, 0.6], [0.9, 0.1, 0]]
        >>> metric = MulticlassROCCurve(num_classes=3, thresholds=4)
        >>> metric(target, preds)
        (array([[0.        , 0.        , 0.        , 1.        ],
                [0.        , 0.33333333, 0.33333333, 1.        ],
                [0.        , 0.        , 0.        , 1.        ]]),
        array([[0. , 0.5, 0.5, 1. ],
                [0. , 1. , 1. , 1. ],
                [0. , 0. , 1. , 1. ]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))
        >>> metric.reset_state()
        >>> target = [[1, 1, 0, 0], [0, 0, 1, 1]]
        >>> preds = [[[0.1, 0.2, 0.7], [0.5, 0.4, 0.1],
        ...         [0.2, 0.3, 0.5], [0.8, 0.1, 0.1]],
        ...         [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1],
        ...         [0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0.  , 0.25, 0.5 , 1.  ],
                [0.  , 0.  , 0.25, 1.  ],
                [0.  , 0.25, 0.5 , 1.  ]]),
        array([[0.  , 0.25, 0.5 , 1.  ],
                [0.  , 0.  , 0.25, 1.  ],
                [0.  , 0.  , 0.  , 0.  ]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))

    """

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the ROC curve from the state variables."""
        # pylint: disable=no-member # attributes are set with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _multiclass_roc_compute(
            state=state, num_classes=self.num_classes, thresholds=self.thresholds
        )


class MultilabelROCCurve(MultilabelPrecisionRecallCurve):
    """Compute the ROC curve for multilabel classification tasks.

    Parameters
    ----------
        num_labels : int
            The number of labels in the dataset.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for binarizing the values of ``preds``.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelROCCurve
        >>> target = [[1, 1, 0], [0, 1, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
        >>> metric = MultilabelROCCurve(num_labels=3, thresholds=4)
        >>> metric(target, preds)
        (array([[0. , 0. , 0. , 1. ],
                [0. , 0. , 0. , 0. ],
                [0. , 0.5, 0.5, 1. ]]),
        array([[0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 0., 0.]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))
        >>> metric.reset_state()
        >>> target = [[[1, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 1, 0]]]
        >>> preds = [[[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
        ...         [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0. , 0. , 0. , 1. ],
                [0. , 0. , 0. , 0. ],
                [0. , 0.5, 0.5, 1. ]]),
        array([[0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 0., 0.]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))

    """

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the ROC curve from the state variables."""
        # pylint: disable=no-member # attributes are set with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _multilabel_roc_compute(
            state=state, num_labels=self.num_labels, thresholds=self.thresholds
        )


class ROCCurve(Metric):
    """Compute the ROC curve for different types of classification tasks.

    Parameters
    ----------
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
            Label to consider as positive for binary classification tasks.
        num_classes : int, optional
            The number of classes in the dataset. Required if ``task`` is
            ``"multiclass"``.
        num_labels : int, optional
            The number of labels in the dataset. Required if ``task`` is
            ``"multilabel"``.

    Examples (binary)
    -----------------
        >>> from cyclops.evaluation.metrics import ROCCurve
        >>> target = [0, 0, 1, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> metric = ROCCurve(task="binary", thresholds=None)
        >>> metric(target, preds)
        (array([0. , 0. , 0.5, 0.5, 1. ]),
        array([0. , 0.5, 0.5, 1. , 1. ]),
        array([1.  , 0.8 , 0.4 , 0.35, 0.1 ]))
        >>> metric.reset_state()
        >>> target = [[1, 1, 0, 0], [0, 0, 1, 1]]
        >>> preds = [[0.1, 0.2, 0.3, 0.4], [0.6, 0.5, 0.4, 0.3]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.  , 1.  ]),
        array([0.  , 0.  , 0.  , 0.25, 0.5 , 0.75, 1.  ]),
        array([1. , 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]))

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import ROCCurve
        >>> target = [[1, 1, 0], [0, 1, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
        >>> metric = ROCCurve(task="multiclass", num_classes=3, thresholds=4)
        >>> metric(target, preds)
        (array([[0. , 0. , 0. , 1. ],
                [0. , 0. , 0. , 0. ],
                [0. , 0.5, 0.5, 1. ]]),
        array([[0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 0., 0.]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))
        >>> metric.reset_state()
        >>> target = [[[1, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 1, 0]]]
        >>> preds = [[[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
        ...         [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0. , 0. , 0. , 1. ],
                [0. , 0. , 0. , 0. ],
                [0. , 0.5, 0.5, 1. ]]),
        array([[0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 0., 0.]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))

    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import ROCCurve
        >>> target = [[1, 1, 0], [0, 1, 0]]
        >>> preds = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
        >>> metric = ROCCurve(task="multilabel", num_labels=3, thresholds=4)
        >>> metric(target, preds)
        (array([[0. , 0. , 0. , 1. ],
                [0. , 0. , 0. , 0. ],
                [0. , 0.5, 0.5, 1. ]]),
        array([[0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 0., 0.]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))
        >>> metric.reset_state()
        >>> target = [[[1, 1, 0], [0, 1, 0]], [[1, 1, 0], [0, 1, 0]]]
        >>> preds = [[[0.1, 0.9, 0.8], [0.05, 0.95, 0]],
        ...         [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0. , 0. , 0. , 1. ],
                [0. , 0. , 0. , 0. ],
                [0. , 0.5, 0.5, 1. ]]),
        array([[0., 0., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 0., 0.]]),
        array([1.        , 0.66666667, 0.33333333, 0.        ]))

    """

    def __new__(  # type: ignore # mypy expects a subclass of ROCCurve
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Union[int, List[float], np.ndarray] = None,
        pos_label: int = 1,
        num_classes: int = None,
        num_labels: int = None,
    ) -> Metric:
        """Create a task-specific instance of the ROC curve metric."""
        if task == "binary":
            return BinaryROCCurve(thresholds=thresholds, pos_label=pos_label)
        if task == "multiclass":
            assert isinstance(
                num_classes, int
            ), "Number of classes must be a positive integer."
            return MulticlassROCCurve(num_classes=num_classes, thresholds=thresholds)
        if task == "multilabel":
            assert isinstance(
                num_labels, int
            ), "Number of labels must be a positive integer."
            return MultilabelROCCurve(num_labels=num_labels, thresholds=thresholds)
        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}"
        )
