"""Plotting functions."""

from typing import Optional, Union

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyclops.processors.column_names import (
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
)
from cyclops.utils.common import to_list

PLOT_HEIGHT = 520


def plot_timeline(
    events: pd.DataFrame,
    return_fig: bool = False,
) -> Union[plotly.graph_objs.Figure, None]:
    """Plot timeline of patient events for an encounter.

    Parameters
    ----------
    events: pandas.DataFrame
        Event data to plot.
    return_fig: bool, optional
        Return fig.

    """
    fig = px.strip(
        events,
        x=EVENT_TIMESTAMP,
        y=EVENT_NAME,
        color=EVENT_CATEGORY,
        hover_data=[EVENT_VALUE],
    )

    fig.update_layout(
        title="Timeline Visualization",
        autosize=False,
        height=PLOT_HEIGHT,
    )

    if return_fig:
        return fig
    fig.show()

    return None


def plot_histogram(
    features: pd.DataFrame, name: Optional[list] = None, return_fig: bool = False
) -> Union[plotly.graph_objs.Figure, None]:
    """Plot histogram of static features.

    Plots the histogram of static features over all encounters.
    If 'name' is not specified, then all available features are
    plotted.

    Parameters
    ----------
    features: pandas.DataFrame
        Static features for multiple encounters.
    name: str
        Name of feature to plot over all encounters.

    Raises
    ------
    ValueError
        If any of the provided names is not present as a column in features data,
        error is raised.

    """
    if name not in features:
        raise ValueError(f"Provided feature {name} not present in features data!")
    fig = px.histogram(features[name], marginal="rug")
    fig.update_layout(
        title="Static Feature Visualization",
        autosize=False,
        height=PLOT_HEIGHT,
    )
    if return_fig:
        return fig
    fig.show()

    return None


def plot_temporal_features(
    features: pd.DataFrame,
    names: list = None,
    return_fig: bool = False,
) -> Union[plotly.graph_objs.Figure, None]:
    """Plot temporal features.

    Plots a few time-series features (passed for a specific encounter).
    If 'names' is not specified, then all available features are
    plotted. Supports a maximum of 7 features to plot,

    Parameters
    ----------
    features: pandas.DataFrame
        Temporal features for a single encounter.
    names: list, optional
        Names of features to plot for the given encounter.

    Raises
    ------
    ValueError
        If any of the provided names is not present as a column in features data,
        error is raised.

    """
    if names is None:
        feature_names = features.columns
    feature_names = to_list(names)
    for name in feature_names:
        if name not in features:
            raise ValueError(f"Provided feature {name} not present in features data!")
    num_timesteps = len(features.index)
    if feature_names:
        num_plot_rows = len(feature_names)
    else:
        num_plot_rows = 1
    fig = make_subplots(rows=num_plot_rows, cols=1, x_title="timestep", y_title="value")
    for idx, name in enumerate(feature_names):
        fig.add_trace(
            go.Scatter(x=features.index, y=features[name], name=name),
            row=idx + 1,
            col=1,
        )
    if len(feature_names) > 2:
        fig = fig.update_layout(
            title="Temporal Feature Visualization",
            autosize=False,
            height=int(PLOT_HEIGHT / 2 * len(feature_names)),
        )
    else:
        fig = fig.update_layout(
            title="Temporal Feature Visualization", autosize=False, height=PLOT_HEIGHT
        )
    fig.add_vrect(
        x0=0,
        x1=6,
        annotation_text="ER",
        annotation_position="top left",
        fillcolor="orange",
        opacity=0.25,
        line_width=0,
    )
    fig.add_vrect(
        x0=6,
        x1=num_timesteps,
        annotation_text="IP",
        annotation_position="top left",
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )
    if return_fig:
        return fig
    fig.show()

    return None
