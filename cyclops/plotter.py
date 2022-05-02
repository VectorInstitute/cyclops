"""Plotting functions."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_histogram(features: pd.DataFrame, name: Optional[list] = None) -> None:
    """Plot histogram of static features.

    Plots the histogram of static features over all encounters.
    If 'name' is not specified, then all available features are
    plotted.

    Parameters
    ----------
    features: pandas.DataFrame
        All available static features.
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
    fig = px.histogram(features[name])
    fig.update_layout(
        title="Static Feature Visualization",
        autosize=False,
        width=1000,
        height=500,
    )
    fig.show()


def plot_temporal_features(
    features: pd.DataFrame, encounter_id: int, names: Optional[list] = None
) -> None:
    """Plot temporal features.

    Plots a few time-series features for specified encounter.
    If 'names' is not specified, then all available features are
    plotted. Supports a maximum of 7 features to plot,

    Parameters
    ----------
    features: pandas.DataFrame
        All available temporal features.
    encounter_id: int
        Encounter ID.
    names: list, optional
        Names of features to plot for the given encounter.

    Raises
    ------
    ValueError
        If any of the provided names is not present as a column in features data,
        error is raised.

    """
    if names is None:
        names = features.columns
    for name in names:
        if name not in features:
            raise ValueError(f"Provided feature {name} not present in features data!")

    features_encounter = features.loc[encounter_id][names]
    num_timesteps = len(features_encounter.index)
    fig = make_subplots(rows=len(names), cols=1, x_title="timestep", y_title="value")

    for idx, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=features_encounter.index, y=features_encounter[name], name=name
            ),
            row=idx + 1,
            col=1,
        )

    fig.update_layout(
        title="Temporal Feature Visualization",
        autosize=False,
        width=1000,
        height=int(240 * len(names)),
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
    fig.show()
