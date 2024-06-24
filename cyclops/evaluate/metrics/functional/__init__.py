"""Evaluation metrics functional package."""

from cyclops.evaluate.metrics.functional.accuracy import (  # noqa: F401
    accuracy,
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from cyclops.evaluate.metrics.functional.auroc import (
    auroc,  # noqa: F401
    binary_auroc,
    multiclass_auroc,
    multilabel_auroc,
)
from cyclops.evaluate.metrics.functional.average_precision import (  # noqa: F401
    average_precision,
    binary_average_precision,
)
from cyclops.evaluate.metrics.functional.f_beta import (  # noqa: F401
    binary_f1_score,
    binary_fbeta_score,
    f1_score,
    fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from cyclops.evaluate.metrics.functional.precision_recall import (  # noqa: F401
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
    precision,
    recall,
)
from cyclops.evaluate.metrics.functional.precision_recall_curve import (  # noqa: F401
    PRCurve,
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
    precision_recall_curve,
)
from cyclops.evaluate.metrics.functional.roc import (  # noqa: F401
    ROCCurve,
    binary_roc_curve,
    multiclass_roc_curve,
    multilabel_roc_curve,
    roc_curve,
)
from cyclops.evaluate.metrics.functional.sensitivity import (  # noqa: F401
    binary_sensitivity,
    multiclass_sensitivity,
    multilabel_sensitivity,
    sensitivity,
)
from cyclops.evaluate.metrics.functional.specificity import (  # noqa: F401
    binary_specificity,
    multiclass_specificity,
    multilabel_specificity,
    specificity,
)
from cyclops.evaluate.metrics.functional.stat_scores import (  # noqa: F401
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
    stat_scores,
)


__all__ = [
    "accuracy",
    "binary_accuracy",
    "multiclass_accuracy",
    "multilabel_accuracy",
    "auroc",
    "binary_auroc",
    "multiclass_auroc",
    "multilabel_auroc",
    "average_precision",
    "binary_average_precision",
    "binary_f1_score",
    "binary_fbeta_score",
    "f1_score",
    "fbeta_score",
    "multiclass_f1_score",
    "multiclass_fbeta_score",
    "multilabel_f1_score",
    "multilabel_fbeta_score",
    "binary_precision",
    "binary_recall",
    "multiclass_precision",
    "multiclass_recall",
    "multilabel_precision",
    "multilabel_recall",
    "precision",
    "recall",
    "PRCurve",
    "binary_precision_recall_curve",
    "multiclass_precision_recall_curve",
    "multilabel_precision_recall_curve",
    "precision_recall_curve",
    "ROCCurve",
    "binary_roc_curve",
    "multiclass_roc_curve",
    "multilabel_roc_curve",
    "roc_curve",
    "binary_sensitivity",
    "multiclass_sensitivity",
    "multilabel_sensitivity",
    "sensitivity",
    "binary_specificity",
    "multiclass_specificity",
    "multilabel_specificity",
    "specificity",
    "binary_stat_scores",
    "multiclass_stat_scores",
    "multilabel_stat_scores",
    "stat_scores",
]
