"""Evaluation metrics package."""

from cyclops.evaluate.metrics.accuracy import (
    Accuracy,  # noqa: F401
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
)
from cyclops.evaluate.metrics.auroc import (
    AUROC,
    BinaryAUROC,  # noqa: F401
    MulticlassAUROC,
    MultilabelAUROC,
)
from cyclops.evaluate.metrics.average_precision import (
    BinaryAveragePrecision,
)
from cyclops.evaluate.metrics.f_beta import (
    BinaryF1Score,  # noqa: F401
    BinaryFbetaScore,
    F1Score,
    FbetaScore,
    MulticlassF1Score,
    MulticlassFbetaScore,
    MultilabelF1Score,
    MultilabelFbetaScore,
)
from cyclops.evaluate.metrics.factory import create_metric  # noqa: F401
from cyclops.evaluate.metrics.metric import MetricCollection  # noqa: F401
from cyclops.evaluate.metrics.precision_recall import (  # noqa: F401
    BinaryPrecision,
    BinaryRecall,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
    Precision,
    Recall,
)
from cyclops.evaluate.metrics.precision_recall_curve import (  # noqa: F401
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
    PrecisionRecallCurve,
)
from cyclops.evaluate.metrics.roc import (
    BinaryROCCurve,  # noqa: F401
    MulticlassROCCurve,
    MultilabelROCCurve,
    ROCCurve,
)
from cyclops.evaluate.metrics.sensitivity import (  # noqa: F401
    BinarySensitivity,
    MulticlassSensitivity,
    MultilabelSensitivity,
    Sensitivity,
)
from cyclops.evaluate.metrics.specificity import (  # noqa: F401
    BinarySpecificity,
    MulticlassSpecificity,
    MultilabelSpecificity,
    Specificity,
)
from cyclops.evaluate.metrics.stat_scores import (  # noqa: F401
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
    StatScores,
)


__all__ = [
    "create_metric",
    "MetricCollection",
    "Accuracy",
    "AUROC",
    "BinaryAccuracy",
    "BinaryAUROC",
    "BinaryAveragePrecision",
    "BinaryPrecisionRecallCurve",
    "MulticlassPrecisionRecallCurve",
    "MultilabelPrecisionRecallCurve",
    "PrecisionRecallCurve",
    "BinaryF1Score",
    "BinaryFbetaScore",
    "BinaryPrecision",
    "BinaryRecall",
    "BinaryROCCurve",
    "BinarySensitivity",
    "BinarySpecificity",
    "F1Score",
    "FbetaScore",
    "MulticlassAccuracy",
    "MulticlassAUROC",
    "MulticlassAveragePrecision",
    "MulticlassF1Score",
    "MulticlassFbetaScore",
    "MulticlassPrecision",
    "MulticlassRecall",
    "MulticlassROCCurve",
    "MulticlassSensitivity",
    "MulticlassSpecificity",
    "MultilabelAccuracy",
    "MultilabelAUROC",
    "MultilabelAveragePrecision",
    "MultilabelF1Score",
    "MultilabelFbetaScore",
    "MultilabelPrecision",
    "MultilabelRecall",
    "MultilabelROCCurve",
    "MultilabelSensitivity",
    "MultilabelSpecificity",
    "Precision",
    "Recall",
    "ROCCurve",
    "Sensitivity",
    "Specificity",
    "StatScores",
    "BinaryStatScores",
    "MulticlassStatScores",
    "MultilabelStatScores",
]
