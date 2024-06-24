"""Functional metrics for evaluating model performance."""

from cyclops.evaluate.metrics.experimental.functional.accuracy import (
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from cyclops.evaluate.metrics.experimental.functional.auroc import (
    binary_auroc,
    multiclass_auroc,
    multilabel_auroc,
)
from cyclops.evaluate.metrics.experimental.functional.average_precision import (
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from cyclops.evaluate.metrics.experimental.functional.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from cyclops.evaluate.metrics.experimental.functional.f_score import (
    binary_f1_score,
    binary_fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from cyclops.evaluate.metrics.experimental.functional.mae import mean_absolute_error
from cyclops.evaluate.metrics.experimental.functional.mape import (
    mean_absolute_percentage_error,
)
from cyclops.evaluate.metrics.experimental.functional.matthews_corr_coef import (
    binary_mcc,
    multiclass_mcc,
    multilabel_mcc,
)
from cyclops.evaluate.metrics.experimental.functional.mse import mean_squared_error
from cyclops.evaluate.metrics.experimental.functional.negative_predictive_value import (
    binary_npv,
    multiclass_npv,
    multilabel_npv,
)
from cyclops.evaluate.metrics.experimental.functional.precision_recall import (
    binary_ppv,
    binary_precision,
    binary_recall,
    binary_tpr,
    multiclass_ppv,
    multiclass_precision,
    multiclass_recall,
    multiclass_tpr,
    multilabel_ppv,
    multilabel_precision,
    multilabel_recall,
    multilabel_tpr,
)
from cyclops.evaluate.metrics.experimental.functional.precision_recall_curve import (
    PRCurve,
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from cyclops.evaluate.metrics.experimental.functional.roc import (
    ROCCurve,
    binary_roc,
    multiclass_roc,
    multilabel_roc,
)
from cyclops.evaluate.metrics.experimental.functional.smape import (
    symmetric_mean_absolute_percentage_error,
)
from cyclops.evaluate.metrics.experimental.functional.specificity import (
    binary_specificity,
    binary_tnr,
    multiclass_specificity,
    multiclass_tnr,
    multilabel_specificity,
    multilabel_tnr,
)
from cyclops.evaluate.metrics.experimental.functional.wmape import (
    weighted_mean_absolute_percentage_error,
)


__all__ = [
    "binary_accuracy",
    "multiclass_accuracy",
    "multilabel_accuracy",
    "binary_auroc",
    "multiclass_auroc",
    "multilabel_auroc",
    "binary_average_precision",
    "multiclass_average_precision",
    "multilabel_average_precision",
    "binary_confusion_matrix",
    "multiclass_confusion_matrix",
    "multilabel_confusion_matrix",
    "binary_f1_score",
    "binary_fbeta_score",
    "multiclass_f1_score",
    "multiclass_fbeta_score",
    "multilabel_f1_score",
    "multilabel_fbeta_score",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "binary_mcc",
    "multiclass_mcc",
    "multilabel_mcc",
    "binary_npv",
    "multiclass_npv",
    "multilabel_npv",
    "binary_ppv",
    "binary_precision",
    "binary_recall",
    "binary_tpr",
    "multiclass_ppv",
    "multiclass_precision",
    "multiclass_recall",
    "multiclass_tpr",
    "multilabel_ppv",
    "multilabel_precision",
    "multilabel_recall",
    "multilabel_tpr",
    "PRCurve",
    "binary_precision_recall_curve",
    "multiclass_precision_recall_curve",
    "multilabel_precision_recall_curve",
    "ROCCurve",
    "binary_roc",
    "multiclass_roc",
    "multilabel_roc",
    "symmetric_mean_absolute_percentage_error",
    "binary_specificity",
    "binary_tnr",
    "multiclass_specificity",
    "multiclass_tnr",
    "multilabel_specificity",
    "multilabel_tnr",
    "weighted_mean_absolute_percentage_error",
]
