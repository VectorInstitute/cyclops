from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_pretty_confusion_matrix(confusion_matrix):
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        np.eye(2),
        annot=confusion_matrix,
        fmt="g",
        annot_kws={"size": 50},
        cmap=sns.color_palette(["tomato", "palegreen"], as_cmap=True),
        cbar=False,
        yticklabels=["True", "False"],
        xticklabels=["True", "False"],
        ax=ax,
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(labelsize=20, length=0)

    ax.set_title("Confusion Matrix for Test Set", size=24, pad=20)
    ax.set_xlabel("Predicted Values", size=20)
    ax.set_ylabel("Actual Values", size=20)

    additional_texts = [
        "(True Positive)",
        "(False Negative)",
        "(False Positive)",
        "(True Negative)",
    ]
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(
            *text_elt.get_position(),
            "\n" + additional_text,
            color=text_elt.get_color(),
            ha="center",
            va="top",
            size=24
        )
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix, class_names):
    confusion_matrix = (
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    )

    layout = {
        "title": "Confusion Matrix",
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "Real value"},
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=class_names,
            y=class_names,
            hoverongaps=False,
            colorscale="Greens",
        ),
        layout=layout,
    )
    fig.update_layout(height=512, width=1024)
    fig.show()


def plot_auroc_across_timesteps(
    y_pred_labels,
    y_test_labels,
):
    num_timesteps = y_pred_labels.shape[1]
    auroc_timesteps = []
    for i in range(num_timesteps):
        labels = y_test_labels[:, i]
        pred_vals = y_pred_values[:, i]
        preds = y_pred_labels[:, i]
        pred_vals = pred_vals[labels != -1]
        preds = preds[labels != -1]
        labels = labels[labels != -1]
        pred_metrics = print_metrics_binary(labels, pred_vals, preds, verbose=False)
        auroc_timesteps.append(pred_metrics["auroc"])

    print(auroc_timesteps)

    prediction_hours = list(range(24, 168, 24))
    fig = go.Figure(
        data=[go.Bar(x=prediction_hours, y=auroc_timesteps, name="model confidence")]
    )

    fig.update_xaxes(tickvals=prediction_hours)
    fig.update_yaxes(range=[min(auroc_timesteps) - 0.05, max(auroc_timesteps) + 0.05])

    fig.update_layout(
        title="AUROC split by no. of hours after admission",
        autosize=False,
        xaxis_title="No. of hours after admission",
    )


def plot_risk_mortality(predictions, labels=None):
    prediction_hours = list(range(24, 168, 24))
    is_mortality = labels == 1
    after_mortality = labels == -1
    label_h = -0.2
    fig = go.Figure(
        data=[
            go.Scatter(
                mode="markers",
                x=prediction_hours,
                y=[label_h for x in prediction_hours],
                line=dict(color="Black"),
                name="low risk of mortality label",
                marker=dict(color="Green", size=20, line=dict(color="Black", width=2)),
            ),
            go.Scatter(
                mode="markers",
                x=[prediction_hours[i] for i, v in enumerate(is_mortality) if v],
                y=[label_h for _, v in enumerate(is_mortality) if v],
                line=dict(color="Red"),
                name="high risk of mortality label",
                marker=dict(color="Red", size=20, line=dict(color="Black", width=2)),
            ),
            go.Scatter(
                mode="markers",
                x=[prediction_hours[i] for i, v in enumerate(after_mortality) if v],
                y=[label_h for _, v in enumerate(after_mortality) if v],
                line=dict(color="Grey"),
                name="post mortality label",
                marker=dict(color="Grey", size=20, line=dict(color="Black", width=2)),
            ),
            go.Bar(
                x=prediction_hours,
                y=predictions,
                marker_color="Red",
                name="model confidence",
            ),
        ]
    )
    fig.update_yaxes(range=[label_h, 1])
    fig.update_xaxes(tickvals=prediction_hours)
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black")

    fig.add_hline(y=0.5)

    fig.update_layout(
        title="Model output visualization",
        autosize=False,
        xaxis_title="No. of hours after admission",
        yaxis_title="Model confidence",
    )

    return fig


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
