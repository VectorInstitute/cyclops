"""Cyclops Visualizer Application."""


import logging
import os

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from css import CONTENT_STYLE, SIDEBAR_LIST_STYLE, SIDEBAR_STYLE, TEXT_ALIGN_CENTER
from dash import Input, Output, State, dcc, html
from flask_caching import Cache

from codebase_ops import get_log_file_path
from cyclops.plotter import plot_histogram, plot_temporal_features, plot_timeline
from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.processors.constants import STATIC, TEMPORAL
from cyclops.utils.log import setup_logging

CACHE_TIMEOUT = 3000
EVALUATION = "evaluation"
TIMELINE = "timeline"
FEATURE_STORE = "feature_store"
DIRECT_LOAD = "direct_load"


dir_path = os.path.dirname(os.path.abspath(__file__))
temporal_features = pd.read_parquet(
    os.path.join(dir_path, "test_features_temporal.gzip")
)
static_features = pd.read_parquet(os.path.join(dir_path, "test_features_static.gzip"))
encounters = temporal_features.index.get_level_values(0)
num_encounters = len(encounters)

events = pd.read_parquet(os.path.join(dir_path, "events.gzip"))
encounters_events = list(events[ENCOUNTER_ID].unique())[0:80]
num_encounters_events = len(encounters_events)


# Initialize app.
app = dash.Dash(
    external_stylesheets=[dbc.themes.SIMPLEX], suppress_callback_exceptions=True
)
app.title = "cyclops.visualizer"
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
server = app.server
cache = Cache(
    server,
    config={
        "DEBUG": True,
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": CACHE_TIMEOUT,
    },
)

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


sidebar = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Br(),
        html.H5("Mode"),
        dcc.RadioItems(
            id="view-mode",
            options=[
                {"label": "Encounter Timeline", "value": TIMELINE},
                {"label": "Static Features", "value": STATIC},
                {"label": "Temporal Features", "value": TEMPORAL},
                {"label": "Evaluation", "value": EVALUATION},
            ],
            value=TIMELINE,
            labelStyle=SIDEBAR_LIST_STYLE,
        ),
        html.Br(),
        html.Br(),
        html.H5("Data Sources"),
        dcc.RadioItems(
            id="data-source",
            options=[
                {"label": "Feature Store", "value": FEATURE_STORE},
                {"label": "Direct Load", "value": DIRECT_LOAD},
            ],
            value=DIRECT_LOAD,
            labelStyle=SIDEBAR_LIST_STYLE,
        ),
        html.Br(),
        html.Br(),
    ],
    style=SIDEBAR_STYLE,
)


timeseries_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="temporal-features"),
                    style={"overflowY": "scroll", "height": 500},
                ),
                html.Br(),
                dcc.Dropdown(
                    id="temporal-features-dropdown",
                    value=temporal_features.columns[0],
                    options=[
                        {"label": name, "value": name}
                        for name in temporal_features.columns
                    ],
                    multi=True,
                    searchable=True,
                ),
                html.Div(id="encounter-caption"),
                dcc.Slider(
                    id="encounter-slider",
                    min=0,
                    max=(num_encounters - 1),
                    step=1,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        )
    ],
    style=TEXT_ALIGN_CENTER,
)


timeline_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="timeline"),
                    style={"overflowY": "scroll", "height": 500},
                ),
                html.Br(),
                html.Div(id="encounter-events-caption"),
                dcc.Slider(
                    id="encounter-events-slider",
                    min=0,
                    max=(num_encounters_events - 1),
                    step=1,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        )
    ],
    style=TEXT_ALIGN_CENTER,
)


static_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="static-features"),
                    style={"overflowY": "scroll", "height": 500},
                ),
                dcc.Dropdown(
                    id="static-features-dropdown",
                    value=static_features.columns[0],
                    options=[
                        {"label": name, "value": name}
                        for name in static_features.columns
                    ],
                    multi=False,
                    searchable=True,
                ),
                dcc.Slider(
                    id="encounter-slider",
                    min=0,
                    max=(num_encounters - 1),
                    step=1,
                    value=0,
                ),
            ]
        )
    ],
    style=TEXT_ALIGN_CENTER,
)


offcanvas_sidebar = html.Div(
    [
        dbc.Button(
            "Options",
            id="options",
            n_clicks=0,
        ),
        dbc.Offcanvas(
            sidebar,
            id="offcanvas-scrollable",
            scrollable=True,
            title="Options",
            is_open=False,
        ),
    ]
)


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Br(),
        html.Div(
            [
                html.Img(src=app.get_asset_url("vector_logo.png"), height=32),
                dcc.Markdown("""# cyclops visualizer"""),
            ],
            style=TEXT_ALIGN_CENTER,
        ),
        html.Hr(),
        offcanvas_sidebar,
        dcc.Loading(
            id="loading-view",
            type="default",
            children=html.Div(id="update-view-mode"),
        ),
        html.Br(),
        html.Div(
            [
                dbc.Button("previous", id="btn-prev-sample", color="secondary"),
                dbc.Button(
                    "next",
                    id="btn-next-sample",
                    color="secondary",
                    style={"margin-left": "15px"},
                ),
            ],
            style=TEXT_ALIGN_CENTER,
        ),
    ],
    style=CONTENT_STYLE,
)


# Callbacks.
@app.callback(
    dash.dependencies.Output("update-view-mode", "children"),
    dash.dependencies.Input("view-mode", "value"),
)
def update_view(view_mode):
    """Update main view."""
    if view_mode == TEMPORAL:
        return timeseries_layout
    if view_mode == STATIC:
        return static_layout
    if view_mode == EVALUATION:
        return timeline_layout
    if view_mode == TIMELINE:
        return timeline_layout

    return timeline_layout


@app.callback(
    [
        dash.dependencies.Output("timeline", "figure"),
        dash.dependencies.Output("encounter-events-caption", "children"),
    ],
    {
        **{"index": dash.dependencies.Input("encounter-events-slider", "value")},
    },
)
def update_timeline_plot(index):
    """Update timeline plot."""
    events_encounter = events.loc[events[ENCOUNTER_ID] == encounters_events[index]]
    return [
        plot_timeline(events_encounter, return_fig=True),
        f'Encounter ID: "{encounters_events[index]}"',
    ]


@app.callback(
    [
        dash.dependencies.Output("temporal-features", "figure"),
        dash.dependencies.Output("encounter-caption", "children"),
    ],
    {
        **{"index": dash.dependencies.Input("encounter-slider", "value")},
        **{
            "feature_names": dash.dependencies.Input(
                "temporal-features-dropdown", "value"
            )
        },
    },
)
def update_temporal_features_plot(index, feature_names):
    """Update timeseries plot."""
    features_encounter = temporal_features.loc[encounters[index]]
    return [
        plot_temporal_features(
            features_encounter, names=feature_names, return_fig=True
        ),
        'Encounter ID: "{encounters[index]}"',
    ]


@app.callback(
    [
        dash.dependencies.Output("static-features", "figure"),
    ],
    {
        **{
            "feature_names": dash.dependencies.Input(
                "static-features-dropdown", "value"
            )
        },
    },
)
def update_histogram_plot(feature_names):
    """Update histogram plot of static features."""
    return [
        plot_histogram(static_features, name=feature_names, return_fig=True),
    ]


@app.callback(
    Output("offcanvas-scrollable", "is_open"),
    Input("options", "n_clicks"),
    State("offcanvas-scrollable", "is_open"),
)
def toggle_offcanvas_scrollable(num_clicks, is_open):
    """Toggle offcanvas scrolling."""
    if num_clicks:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port="8504")
