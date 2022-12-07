"""Utilities for static baseline models."""
from sklearn.metrics import confusion_matrix

from .gbt import fit_gbt
from .lr import fit_lr
from .mlp import fit_mlp
from .rf import fit_rf


def run_model(model_name, X, y, X_val, y_val):
    """Choose and run a model on the data and return the best model."""
    if model_name == "mlp":
        best_model = fit_mlp(X, y, X_val, y_val)
    elif model_name == "lr":
        best_model = fit_lr(X, y, X_val, y_val)
    elif model_name == "rf":
        best_model = fit_rf(X, y, X_val, y_val)
    elif model_name == "xgb":
        best_model = fit_gbt(X, y, X_val, y_val)
    return best_model


def compute_threshold_metric(y_true, pred_prob, threshold):
    """Threshold metrics for binary prediction tasks."""
    y_pred = 1 * (pred_prob > threshold)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    metric_dict = {}

    # Sensitivity, hit rate, recall, or true positive rate
    metric_dict["sensitivity"] = TP / (TP + FN)
    # Specificity or true negative rate
    metric_dict["specificity"] = TN / (TN + FP)
    # Precision or positive predictive value
    metric_dict["ppv"] = TP / (TP + FP)
    # Negative predictive value
    metric_dict["npv"] = TN / (TN + FN)
    # Fall out or false positive rate
    metric_dict["fpr"] = FP / (FP + TN)
    # False negative rate
    metric_dict["fnr"] = FN / (TP + FN)
    # False discovery rate
    metric_dict["fdr"] = FP / (TP + FP)

    return metric_dict
