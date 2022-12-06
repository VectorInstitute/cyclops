"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from cyclops.models.util import metrics_binary

# pylint: disable=invalid-name


def plot_pretty_confusion_matrix(confusion_matrix: np.ndarray) -> None:
    """Plot pretty confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        confusion matrix

    """
    sns.set(style="white")
    _, ax = plt.subplots(figsize=(9, 6))
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


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: list) -> go.Figure:
    """Plot confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        confusion matrix
    class_names : list
        data class names

    Returns
    -------
    go.Figure
        plot figure

    """
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
    return fig


def plot_auroc_across_timesteps(
    y_pred_values: np.ndarray,
    y_pred_labels: np.ndarray,
    y_test_labels: np.ndarray,
) -> go.Figure:
    """Plot AUC_ROC across timesteps.

    Parameters
    ----------
    y_pred_values : np.ndarray
        prediction values
    y_pred_labels : np.ndarray
        prediction labels
    y_test_labels : np.ndarray
        data labels

    Returns
    -------
    go.Figure
        plot figures

    """
    num_timesteps = y_pred_labels.shape[1]
    auroc_timesteps = []
    for i in range(num_timesteps):
        labels = y_test_labels[:, i]
        pred_vals = y_pred_values[:, i]
        preds = y_pred_labels[:, i]
        pred_vals = pred_vals[labels != -1]
        preds = preds[labels != -1]
        labels = labels[labels != -1]
        pred_metrics = metrics_binary(labels, pred_vals, preds, verbose=False)
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
    return fig


def plot_risk_mortality(predictions: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Plot risk of mortality as predicted.

    Parameters
    ----------
    predictions : np.ndarray
        mortality predictions
    labels : np.ndarray
        mortality labels, by default None

    Returns
    -------
    go.Figure
        plot figure

    """
    prediction_hours = list(range(24, 168, 24))
    is_mortality = labels == 1
    after_discharge = labels == -1
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
                x=[prediction_hours[i] for i, v in enumerate(after_discharge) if v],
                y=[label_h for _, v in enumerate(after_discharge) if v],
                line=dict(color="Grey"),
                name="post discharge label",
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
