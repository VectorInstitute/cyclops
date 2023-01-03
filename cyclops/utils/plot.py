"""Plotting functions."""

from typing import List, Optional, Union

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.axes import SubplotBase
from matplotlib.container import BarContainer
from plotly.subplots import make_subplots

from cyclops.process.column_names import (
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
)
from cyclops.utils.common import to_list

PLOT_HEIGHT = 520


def plot_timeline(
    events: pd.DataFrame,
    timestep_timestamps: Optional[pd.Series] = None,
    return_fig: bool = False,
) -> Union[plotly.graph_objs.Figure, None]:
    """Plot timeline of patient events for an encounter.

    Parameters
    ----------
    events: pandas.DataFrame
        Event data to plot.
    timestep_timestamps: pandas.Series, optional
        Timestamps of timesteps to overlay as vertical lines.
        Useful to see aggregation buckets.
    return_fig: bool, optional
        Return fig.

    """
    fig = px.strip(
        events,
        x=EVENT_TIMESTAMP,
        y=EVENT_NAME,
        color=EVENT_CATEGORY,
        hover_data=[EVENT_VALUE],
        stripmode="group",
    )
    fig.update_layout(
        title="Timeline Visualization",
        autosize=False,
        height=PLOT_HEIGHT,
    )
    if timestep_timestamps:
        for timestep_timestamp in timestep_timestamps:
            fig.add_vline(timestep_timestamp)

    fig = fig.update_layout(
        {
            "plot_bgcolor": "rgba(255, 0, 0, 0.1)",
            "paper_bgcolor": "rgba(192, 192, 192, 0.25)",
        }
    )

    if return_fig:
        return fig

    fig = fig.update_layout(width=PLOT_HEIGHT * 2)
    fig.show()

    return None


def plot_histogram(
    features: pd.DataFrame,
    names: Union[str, list] = None,
    return_fig: bool = False,
    title="Histogram Visualization",
) -> Union[plotly.graph_objs.Figure, None]:
    """Plot histogram of columns.

    Plots the histogram of columns.
    If 'names' is not specified, then all available columns are
    plotted.

    Parameters
    ----------
    features: pandas.DataFrame
        Feature columns.
    names: list, optional
        Names of feature to plot over all encounters.

    """
    if features is None:
        return make_subplots(rows=1, cols=1)
    if names is None:
        feature_names = list(features.columns)
    else:
        feature_names = to_list(names)
    for name in feature_names:
        if name not in features:
            raise ValueError(f"Provided feature {name} not present in features data!")
    if feature_names:
        num_plot_rows = len(feature_names)
    else:
        num_plot_rows = 1
    fig = make_subplots(rows=num_plot_rows, cols=1)

    for idx, name in enumerate(feature_names):
        fig.add_trace(
            go.Histogram(x=features[name], name=name),
            row=idx + 1,
            col=1,
        )
    fig.update_layout(
        title=title,
        autosize=False,
        height=int(PLOT_HEIGHT * len(feature_names)),
    )
    if return_fig:
        return fig
    fig.show()

    return None


def plot_temporal_features(
    features: pd.DataFrame,
    names: Union[str, List] = None,
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
        feature_names = list(features.columns)
    else:
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


def setup_plot(
    plot_handle: SubplotBase,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: list,
):
    """Set some attributes to plot e.g. title, labels and legend.

    Parameters
    ----------
    plot_handle: matplotlib.axes.SubplotBase
        Subplot handle.
    title: str
        Title of plot.
    xlabel: str
        Label for x-axis.
    ylabel: str
        Label for y-axis.
    legend: list
        Legend for different sub-groups.

    """
    plot_handle.title.set_text(title)
    plot_handle.set_xlabel(xlabel, fontsize=20)
    plot_handle.set_ylabel(ylabel, fontsize=20)
    plot_handle.legend(legend, loc=1)


def set_bars_color(bars: BarContainer, color: str):
    """Set color attribute for bars in bar plots.

    Parameters
    ----------
    bars: matplotlib.container.BarContainer
        Bars.
    color: str
        Color.

    """
    for bar_ in bars:
        bar_.set_color(color)
