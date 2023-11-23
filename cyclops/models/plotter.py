"""Plotting functions."""

import numpy as np
import plotly.graph_objects as go


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
                y=[label_h for _ in prediction_hours],
                line={"color": "Black"},
                name="low risk of mortality label",
                marker={
                    "color": "Green",
                    "size": 20,
                    "line": {"color": "Black", "width": 2},
                },
            ),
            go.Scatter(
                mode="markers",
                x=[prediction_hours[i] for i, v in enumerate(is_mortality) if v],
                y=[label_h for _, v in enumerate(is_mortality) if v],
                line={"color": "Red"},
                name="high risk of mortality label",
                marker={
                    "color": "Red",
                    "size": 20,
                    "line": {"color": "Black", "width": 2},
                },
            ),
            go.Scatter(
                mode="markers",
                x=[prediction_hours[i] for i, v in enumerate(after_discharge) if v],
                y=[label_h for _, v in enumerate(after_discharge) if v],
                line={"color": "Grey"},
                name="post discharge label",
                marker={
                    "color": "Grey",
                    "size": 20,
                    "line": {"color": "Black", "width": 2},
                },
            ),
            go.Bar(
                x=prediction_hours,
                y=predictions,
                marker_color="Red",
                name="model confidence",
            ),
        ],
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
