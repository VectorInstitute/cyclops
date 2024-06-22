"""Metrics for arrays that conform to the Python array API standard."""

from cyclops.evaluate.metrics.experimental.accuracy import (
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
)
from cyclops.evaluate.metrics.experimental.auroc import (
    BinaryAUROC,
    MulticlassAUROC,
    MultilabelAUROC,
)
from cyclops.evaluate.metrics.experimental.average_precision import (
    BinaryAveragePrecision,
    MulticlassAveragePrecision,
    MultilabelAveragePrecision,
)
from cyclops.evaluate.metrics.experimental.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from cyclops.evaluate.metrics.experimental.f_score import (
    BinaryF1Score,
    BinaryFBetaScore,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MultilabelF1Score,
    MultilabelFBetaScore,
)
from cyclops.evaluate.metrics.experimental.mae import MeanAbsoluteError
from cyclops.evaluate.metrics.experimental.mape import MeanAbsolutePercentageError
from cyclops.evaluate.metrics.experimental.matthews_corr_coef import (
    BinaryMCC,
    MulticlassMCC,
    MultilabelMCC,
)
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict
from cyclops.evaluate.metrics.experimental.mse import MeanSquaredError
from cyclops.evaluate.metrics.experimental.negative_predictive_value import (
    BinaryNPV,
    MulticlassNPV,
    MultilabelNPV,
)
from cyclops.evaluate.metrics.experimental.precision_recall import (
    BinaryPPV,
    BinaryPrecision,
    BinaryRecall,
    BinarySensitivity,
    BinaryTPR,
    MulticlassPPV,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSensitivity,
    MulticlassTPR,
    MultilabelPPV,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelSensitivity,
    MultilabelTPR,
)
from cyclops.evaluate.metrics.experimental.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from cyclops.evaluate.metrics.experimental.roc import (
    BinaryROC,
    MulticlassROC,
    MultilabelROC,
)
from cyclops.evaluate.metrics.experimental.smape import (
    SymmetricMeanAbsolutePercentageError,
)
from cyclops.evaluate.metrics.experimental.specificity import (
    BinarySpecificity,
    BinaryTNR,
    MulticlassSpecificity,
    MulticlassTNR,
    MultilabelSpecificity,
    MultilabelTNR,
)
from cyclops.evaluate.metrics.experimental.wmape import (
    WeightedMeanAbsolutePercentageError,
)


__all__ = [
    "BinaryAccuracy",
    "MulticlassAccuracy",
    "MultilabelAccuracy",
    "BinaryAUROC",
    "MulticlassAUROC",
    "MultilabelAUROC",
    "BinaryAveragePrecision",
    "MulticlassAveragePrecision",
    "MultilabelAveragePrecision",
    "BinaryConfusionMatrix",
    "MulticlassConfusionMatrix",
    "MultilabelConfusionMatrix",
    "BinaryF1Score",
    "BinaryFBetaScore",
    "MulticlassF1Score",
    "MulticlassFBetaScore",
    "MultilabelF1Score",
    "MultilabelFBetaScore",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "BinaryMCC",
    "MulticlassMCC",
    "MultilabelMCC",
    "MetricDict",
    "BinaryNPV",
    "MulticlassNPV",
    "MultilabelNPV",
    "BinaryPPV",
    "BinaryPrecision",
    "BinaryRecall",
    "BinarySensitivity",
    "BinaryTPR",
    "MulticlassPPV",
    "MulticlassPrecision",
    "MulticlassRecall",
    "MulticlassSensitivity",
    "MulticlassTPR",
    "MultilabelPPV",
    "MultilabelPrecision",
    "MultilabelRecall",
    "MultilabelSensitivity",
    "MultilabelTPR",
    "BinaryPrecisionRecallCurve",
    "MulticlassPrecisionRecallCurve",
    "MultilabelPrecisionRecallCurve",
    "BinaryROC",
    "MulticlassROC",
    "MultilabelROC",
    "SymmetricMeanAbsolutePercentageError",
    "BinarySpecificity",
    "BinaryTNR",
    "MulticlassSpecificity",
    "MulticlassTNR",
    "MultilabelSpecificity",
    "MultilabelTNR",
    "WeightedMeanAbsolutePercentageError",
]
