"""Plotting functions."""

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from cyclops.constants import FEATURES


def plot_static_features(features: pd.DataFrame, names: Optional[list] = None) -> None:
    """Plot historgram of static features.

    Plots the time-series features for specified encounter.
    If 'names' is not specified, then all available features are
    plotted.

    Parameters
    ----------
    features: pandas.DataFrame
        All available temporal features.
    names: list, optional
        Names of features to plot for the given encounter.

    Raises
    ------
    ValueError
        If any of the provided names is not present as a column in features data,
        error is raised.

    """


def plot_temporal_features(
    features: pd.DataFrame, encounter_id: int, names: Optional[list] = None
) -> None:
    """Plot temporal features.

    Plots the time-series features for specified encounter.
    If 'names' is not specified, then all available features are
    plotted.

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
    features_encounter.columns.name = FEATURES

    fig = px.line(
        features_encounter, facet_col=FEATURES, facet_col_wrap=1, markers=True
    )
    fig.update_layout(
        title="Temporal Feature Visualization",
        autosize=False,
        width=1000,
        height=int(240 * len(names)),
    )
    fig.update_yaxes(matches=None)

    fig.add_vrect(
        x0=0,
        x1=6,
        annotation_text="ER",
        annotation_position="top left",
        fillcolor="green",
        opacity=0.25,
        line_width=0,
    )
    fig.show()
