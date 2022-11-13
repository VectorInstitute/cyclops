"""Classes for computing precision-recall curves."""

from typing import List, Literal, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from cyclops.evaluate.metrics.functional.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_update,
    _check_thresholds,
    _format_thresholds,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_update,
)
from cyclops.evaluate.metrics.metric import Metric

# mypy: ignore-errors


class BinaryPrecisionRecallCurve(Metric):
    """Compute precision-recall curve for binary input.

    Parameters
    ----------
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for computing the precision and recall scores.
            If int, then the number of thresholds to use.
            If list or numpy.ndarray, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.
        pos_label : int
            The label of the positive class.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import BinaryPrecisionRecallCurve
        >>> target = [0, 1, 0, 1]
        >>> preds = [0.1, 0.4, 0.35, 0.8]
        >>> metric = BinaryPrecisionRecallCurve(thresholds=3)
        >>> metric(target, preds)
        (array([0.5, 1. , 0. ]), array([1. , 0.5, 0. ]), array([0. , 0.5, 1. ]))
        >>> metric.reset_state()
        >>> target = [[0, 1, 0, 1], [1, 1, 0, 0]]
        >>> preds = [[0.1, 0.4, 0.35, 0.8], [0.6, 0.3, 0.1, 0.7]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([0.5       , 0.66666667, 0.        ]),
        array([1. , 0.5, 0. ]),
        array([0. , 0.5, 1. ]))

    """

    def __init__(
        self,
        thresholds: Union[int, List[float], np.ndarray] = None,
        pos_label: int = 1,
    ) -> None:
        super().__init__()
        _check_thresholds(thresholds)
        thresholds = _format_thresholds(thresholds)
        if thresholds is None:
            self.add_state("preds", default=[])
            self.add_state("target", default=[])
        else:
            self.add_state(
                "confmat", default=np.zeros((len(thresholds), 2, 2), dtype=np.int_)
            )
        self.thresholds = thresholds
        self.pos_label = pos_label

    def update_state(  # pylint: disable=arguments-differ
        self, target: ArrayLike, preds: ArrayLike
    ) -> None:  # type: ignore
        """Update the state of the metric.

        The state is either a list of targets and predictions (if ``thresholds`` is
        ``None``) or a confusion matrix.

        """
        target, preds = _binary_precision_recall_curve_format(
            target=target, preds=preds
        )
        state = _binary_precision_recall_curve_update(
            target=target, preds=preds, thresholds=self.thresholds
        )

        # pylint: disable=no-member # attributes are added with setattr
        if isinstance(state, np.ndarray):
            self.confmat += state
        else:
            self.target.append(state[0])
            self.preds.append(state[1])

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the precision-recall curve from the state."""
        # pylint: disable=no-member # attributes are added with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _binary_precision_recall_curve_compute(
            state=state, thresholds=self.thresholds, pos_label=self.pos_label
        )


class MulticlassPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for multiclass problems.

    Parameters
    ----------
        num_classes : int
            The number of classes in the dataset.
        thresholds : Union[int, List[float], numpy.ndarray], default=None
            Thresholds used for computing the precision and recall scores.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MulticlassPrecisionRecallCurve
        >>> target = [0, 1, 2, 0]
        >>> preds = [[0.1, 0.6, 0.3], [0.05, 0.95, 0.],
        ...          [0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]
        >>> metric = MulticlassPrecisionRecallCurve(num_classes=3, thresholds=3)
        >>> metric(target, preds)
        (array([[0.5       , 0.        , 0.        , 1.        ],
                [0.25      , 0.33333333, 0.        , 1.        ],
                [0.25      , 0.        , 0.        , 1.        ]]),
        array([[1., 0., 0., 0.],
               [1., 1., 0., 0.],
               [1., 0., 0., 0.]]),
        array([0. , 0.5, 1. ]))
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 0], [1, 2, 0, 1]]
        >>> preds = [
        ...     [[0.1, 0.6, 0.3], [0.05, 0.95, 0.], [0.5, 0.3, 0.2], [0.2, 0.5, 0.3]],
        ...     [[0.3, 0.2, 0.5], [0.1, 0.7, 0.2], [0.6, 0.1, 0.3], [0.1, 0.8, 0.1]],
        ... ]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0.375, 0.5  , 0.   , 1.   ],
                [0.375, 0.4  , 0.   , 1.   ],
                [0.25 , 0.   , 0.   , 1.   ]]),
         array([[1.        , 0.33333333, 0.        , 0.        ],
                [1.        , 0.66666667, 0.        , 0.        ],
                [1.        , 0.        , 0.        , 0.        ]]),
         array([0. , 0.5, 1. ]))

    """

    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[float], np.ndarray] = None,
    ) -> None:
        super().__init__()
        _check_thresholds(thresholds)

        thresholds = _format_thresholds(thresholds)
        if thresholds is None:
            self.add_state("preds", default=[])
            self.add_state("target", default=[])
        else:
            self.add_state(
                "confmat",
                default=np.zeros((len(thresholds), num_classes, 2, 2), dtype=np.int_),
            )
        self.thresholds = thresholds
        self.num_classes = num_classes

    def update_state(  # pylint: disable=arguments-differ
        self, target: ArrayLike, preds: ArrayLike
    ) -> None:  # type: ignore
        """Update the state of the metric.

        The state is either a list of targets and predictions (if ``thresholds`` is
        ``None``) or a confusion matrix.

        """
        target, preds = _multiclass_precision_recall_curve_format(
            target=target, preds=preds, num_classes=self.num_classes
        )
        state = _multiclass_precision_recall_curve_update(
            target=target,
            preds=preds,
            thresholds=self.thresholds,
            num_classes=self.num_classes,
        )

        # pylint: disable=no-member # attributes are added with setattr
        if isinstance(state, np.ndarray):
            self.confmat += state
        else:
            self.target.append(state[0])
            self.preds.append(state[1])

    def compute(
        self,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    ]:
        """Compute the precision-recall curve from the state."""
        # pylint: disable=no-member # attributes are added with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _multiclass_precision_recall_curve_compute(
            state=state, thresholds=self.thresholds, num_classes=self.num_classes
        )


class MultilabelPrecisionRecallCurve(Metric):
    """Check and format the multilabel precision-recall curve input/data.

    Parameters
    ----------
        num_labels : int
            The number of labels in the dataset.
        thresholds : int, list of floats or numpy.ndarray of floats, default=None
            Thresholds used for computing the precision and recall scores.
            If int, then the number of thresholds to use.
            If list or array, then the thresholds to use.
            If None, then the thresholds are automatically determined by the
            unique values in ``preds``.

    Examples
    --------
        >>> from cyclops.evaluation.metrics import MultilabelPrecisionRecallCurve
        >>> target = [[0, 1], [1, 0]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> metric = MultilabelPrecisionRecallCurve(num_labels=2, thresholds=3)
        >>> metric(target, preds)
        (array([[0.5, 1. , 0. , 1. ],
                [0.5, 1. , 0. , 1. ]]),
        array([[1., 1., 0., 0.],
                [1., 1., 0., 0.]]),
        array([0. , 0.5, 1. ]))
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]
        >>> preds = [[[0.1, 0.9], [0.8, 0.2]], [[0.2, 0.8], [0.7, 0.3]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0.5, 0.5, 0. , 1. ],
                [0.5, 0.5, 0. , 1. ]]),
        array([[1. , 0.5, 0. , 0. ],
                [1. , 0.5, 0. , 0. ]]),
        array([0. , 0.5, 1. ]))

    """

    def __init__(
        self,
        num_labels: int,
        thresholds: Union[int, List[float], np.ndarray] = None,
    ) -> None:
        super().__init__()

        _check_thresholds(thresholds)
        thresholds = _format_thresholds(thresholds)
        if thresholds is None:
            self.add_state("preds", default=[])
            self.add_state("target", default=[])
        else:
            self.add_state(
                "confmat",
                default=np.zeros((len(thresholds), num_labels, 2, 2), dtype=np.int_),
            )
        self.thresholds = thresholds
        self.num_labels = num_labels

    def update_state(  # pylint: disable=arguments-differ
        self, target: ArrayLike, preds: ArrayLike
    ) -> None:  # type: ignore
        """Update the state of the metric.

        The state is either a list of targets and predictions (if ``thresholds`` is
        ``None``) or a confusion matrix.

        """
        target, preds = _multilabel_precision_recall_curve_format(
            target=target, preds=preds, num_labels=self.num_labels
        )
        state = _multilabel_precision_recall_curve_update(
            target, preds, num_labels=self.num_labels, thresholds=self.thresholds
        )

        # pylint: disable=no-member # attributes are added with setattr
        if isinstance(state, np.ndarray):
            self.confmat += state
        else:
            self.target.append(state[0])
            self.preds.append(state[1])

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the precision-recall curve from the state."""
        # pylint: disable=no-member # attributes are added with setattr
        if self.thresholds is None:
            state = [
                np.concatenate(self.target, axis=0),
                np.concatenate(self.preds, axis=0),
            ]
        else:
            state = self.confmat

        return _multilabel_precision_recall_curve_compute(
            state, thresholds=self.thresholds, num_labels=self.num_labels
        )


class PrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for different classification tasks.

    Parameters
    ----------
        task : Literal["binary", "multiclass", "multilabel"]
            The task for which the precision-recall curve is computed.
        thresholds : int or list of floats or numpy.ndarray of floats, default=None
            Thresholds used for computing the precision and recall scores. If int,
            then the number of thresholds to use. If list or array, then the
            thresholds to use. If None, then the thresholds are automatically
            determined by the sunique values in ``preds``
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
        >>> from cyclops.evaluation.metrics import PrecisionRecallCurve
        >>> target = [1, 1, 1, 0]
        >>> preds = [0.6, 0.2, 0.3, 0.8]
        >>> metric = PrecisionRecallCurve(task="binary", thresholds=None)
        >>> metric(target, preds)
        (array([0.75      , 0.66666667, 0.5       , 0.        , 1.        ]),
        array([1.        , 0.66666667, 0.33333333, 0.        , 0.        ]),
        array([0.2, 0.3, 0.6, 0.8]))
        >>> metric.reset_state()
        >>> target = [[1, 0, 1, 1], [0, 0, 0, 1]]
        >>> preds = [[0.5, 0.4, 0.1, 0.3], [0.9, 0.6, 0.45, 0.8]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        ...
        >>> metric.compute()
        (array([0.5       , 0.42857143, 0.33333333, 0.4       , 0.5       ,
            0.33333333, 0.5       , 0.        , 1.        ]),
        array([1.  , 0.75, 0.5 , 0.5 , 0.5 , 0.25, 0.25, 0.  , 0.  ]),
        array([0.1 , 0.3 , 0.4 , 0.45, 0.5 , 0.6 , 0.8 , 0.9 ]))

    Examples (multiclass)
    ---------------------
        >>> from cyclops.evaluation.metrics import PrecisionRecallCurve
        >>> target = [0, 1, 2, 2]
        >>> preds = [[0.05, 0.95, 0], [0.1, 0.8, 0.1],
        ...         [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]]
        >>> metric = PrecisionRecallCurve(task="multiclass", num_classes=3,
        ...     thresholds=3)
        >>> metric(target, preds)
        (array([[0.25, 0.  , 0.  , 1.  ],
                [0.25, 0.5 , 0.  , 1.  ],
                [0.5 , 1.  , 0.  , 1.  ]]),
        array([[1., 0., 0., 0.],
                [1., 1., 0., 0.],
                [1., 1., 0., 0.]]),
        array([0. , 0.5, 1. ]))
        >>> metric.reset_state()
        >>> target = [[0, 1, 2, 2], [1, 2, 0, 1]]
        >>> preds = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1],
        ...         [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]],
        ...         [[0.05, 0.95, 0], [0.1, 0.8, 0.1],
        ...         [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0.25 , 0.   , 0.   , 1.   ],
                [0.375, 0.5  , 0.   , 1.   ],
                [0.375, 0.5  , 0.   , 1.   ]]),
        array([[1.        , 0.        , 0.        , 0.        ],
                [1.        , 0.66666667, 0.        , 0.        ],
                [1.        , 0.66666667, 0.        , 0.        ]]),
        array([0. , 0.5, 1. ]))


    Examples (multilabel)
    ---------------------
        >>> from cyclops.evaluation.metrics import PrecisionRecallCurve
        >>> target = [[0, 1], [1, 0]]
        >>> preds = [[0.1, 0.9], [0.8, 0.2]]
        >>> metric = PrecisionRecallCurve(task="multilabel", num_labels=2,
        ...     thresholds=3)
        >>> metric(target, preds)
        (array([[0.5, 1. , 0. , 1. ],
                [0.5, 1. , 0. , 1. ]]),
        array([[1., 1., 0., 0.],
                [1., 1., 0., 0.]]),
        array([0. , 0.5, 1. ]))
        >>> metric.reset_state()
        >>> target = [[[0, 1], [1, 0]], [[1, 0], [0, 1]]]
        >>> preds = [[[0.1, 0.9], [0.8, 0.2]],
        ...         [[0.1, 0.9], [0.8, 0.2]]]
        >>> for t, p in zip(target, preds):
        ...     metric.update_state(t, p)
        >>> metric.compute()
        (array([[0.5, 0.5, 0. , 1. ],
                [0.5, 0.5, 0. , 1. ]]),
        array([[1. , 0.5, 0. , 0. ],
                [1. , 0.5, 0. , 0. ]]),
        array([0. , 0.5, 1. ]))

    """

    def __new__(  # type: ignore # mypy expects a subclass of PrecisionRecallCurve
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Union[int, List[float], np.ndarray] = None,
        pos_label: int = 1,
        num_classes: int = None,
        num_labels: int = None,
    ) -> Metric:
        """Create a task-specific instance of the precision-recall curve metric."""
        if task == "binary":
            return BinaryPrecisionRecallCurve(
                thresholds=thresholds, pos_label=pos_label
            )
        if task == "multiclass":
            assert (
                isinstance(num_classes, int) and num_classes > 0
            ), "Number of classes must be a positive integer."
            return MulticlassPrecisionRecallCurve(
                num_classes=num_classes, thresholds=thresholds
            )
        if task == "multilabel":
            assert (
                isinstance(num_labels, int) and num_labels > 0
            ), "Number of labels must be a positive integer."
            return MultilabelPrecisionRecallCurve(
                num_labels=num_labels, thresholds=thresholds
            )
        raise ValueError(
            "Expected argument `task` to be either 'binary', 'multiclass' or "
            f"'multilabel', but got {task}"
        )
