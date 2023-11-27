"""Plotting functions."""

# mypy: ignore-errors

from typing import List, Optional, Union

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyclops.utils.common import to_list


PLOT_HEIGHT = 520


def plot_timeline(
    events: pd.DataFrame,
    event_name: str,
    event_ts: str,
    event_value: str,
    event_category: Optional[str] = None,
    timesteps_ts: Optional[pd.Series] = None,
    return_fig: bool = False,
) -> Union[plotly.graph_objs.Figure, None]:
    """Plot timeline of patient events for an encounter.

    Parameters
    ----------
    events: pandas.DataFrame
        Event data to plot.
    event_name: str
        Name of column with event names.
    event_ts: str
        Name of column with event timestamps.
    event_value: str
        Name of column with event values.
    event_category: str, optional
        Name of column with category of events.
    timesteps_ts: pandas.Series, optional
        Timestamps of timesteps to overlay as vertical lines.
        Useful to see aggregation buckets.
    return_fig: bool, optional
        Return fig.

    """
    fig = px.strip(
        events,
        x=event_ts,
        y=event_name,
        color=event_category,
        hover_data=[event_value],
        stripmode="group",
    )
    fig.update_layout(
        title="Timeline Visualization",
        autosize=True,
    )
    if timesteps_ts is not None:
        for timestep_ts in timesteps_ts:
            fig.add_vline(timestep_ts)

    fig = fig.update_layout(
        {
            "plot_bgcolor": "rgba(255, 0, 0, 0.1)",
            "paper_bgcolor": "rgba(192, 192, 192, 0.25)",
        },
    )

    if return_fig:
        return fig

    fig = fig.update_layout(width=PLOT_HEIGHT * 2)
    fig.show()

    return None


def plot_histogram(
    features: pd.DataFrame,
    names: Optional[Union[str, List[str]]] = None,
    return_fig: bool = False,
    title: str = "Histogram Visualization",
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
    return_fig: bool, optional
        Return fig.
    title: str, optional
        Title of plot.

    """
    if features is None:
        return make_subplots(rows=1, cols=1)
    feature_names = list(features.columns) if names is None else to_list(names)
    for name in feature_names:
        if name not in features:
            raise ValueError(f"Provided feature {name} not present in features data!")
    num_plot_rows = len(feature_names) if feature_names else 1
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
    names: Optional[Union[str, List[str]]] = None,
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
    return_fig: bool, optional
        Return fig.

    Raises
    ------
    ValueError
        If any of the provided names is not present as a column in features data,
        error is raised.

    """
    feature_names = list(features.columns) if names is None else to_list(names)
    for name in feature_names:
        if name not in features:
            raise ValueError(f"Provided feature {name} not present in features data!")
    num_timesteps = len(features.index)
    num_plot_rows = len(feature_names) if feature_names else 1
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
            title="Temporal Feature Visualization",
            autosize=False,
            height=PLOT_HEIGHT,
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
