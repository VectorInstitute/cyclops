from sklearn import metrics
import numpy as np
import pandas as pd


def format_predictions(
    y_pred_values, y_test_labels, y_pred_tags, X_test, last_timestep_only
):
    if last_timestep_only:
        index = X_test.index.unique(level=0)
    else:
        index = X_test.index
    df_result = pd.DataFrame(
        data={
            "y_pred_values": y_pred_values,
            "y_test_labels": y_test_labels,
            "y_pred_labels": y_pred_tags,
        },
        index=index,
    )
    df_result = df_result.sort_index()
    return df_result


def print_metrics_binary(y_test_labels, y_pred_values, y_pred_labels, verbose=1):
    cf = metrics.confusion_matrix(y_test_labels, y_pred_labels)
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_test_labels, y_pred_values)

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(
        y_test_labels, y_pred_values
    )
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
    }
