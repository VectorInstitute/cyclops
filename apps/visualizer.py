"""Cyclops Visualizer Application."""


import logging

import dash
import dash_bootstrap_components as dbc
from css import (
    CONTENT_STYLE,
    SIDEBAR_HEADING_STYLE,
    SIDEBAR_LIST_STYLE,
    SIDEBAR_STYLE,
    TEXT_ALIGN_CENTER,
)
from dash import dcc, html
from flask_caching import Cache

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging

CACHE_TIMEOUT = 3000
TIME_SERIES = "time_series"
EVALUATION = "evaluation"
FEATURE_STORE = "feature_store"
DIRECT_LOAD = "direct_load"


# Initialize app.
app = dash.Dash(
    external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True
)
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
        html.H3("View Modes", className="display-6", style=SIDEBAR_HEADING_STYLE),
        dcc.RadioItems(
            id="view-mode",
            options=[
                {"label": "Time-series", "value": TIME_SERIES},
                {"label": "Evaluation", "value": EVALUATION},
            ],
            value=TIME_SERIES,
            labelStyle=SIDEBAR_LIST_STYLE,
        ),
        html.Br(),
        html.Br(),
        html.H3("Data Sources", className="display-6", style=SIDEBAR_HEADING_STYLE),
        html.Br(),
        html.Br(),
        dcc.RadioItems(
            id="data-source",
            options=[
                {"label": "Feature Store", "value": FEATURE_STORE},
                {"label": "Direct Load", "value": DIRECT_LOAD},
            ],
            value=TIME_SERIES,
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
                    id="frame-id",
                    children="",
                ),
                html.Br(),
                html.Br(),
                dcc.Slider(
                    id="single-image-slider",
                    min=0,
                    max=(100 - 1),
                    step=1,
                    value=0,
                ),
            ]
        )
    ],
    style=TEXT_ALIGN_CENTER,
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
        sidebar,
        dcc.Loading(
            id="loading-view",
            type="default",
            children=html.Div(id="update-view-mode"),
        ),
        html.Br(),
        html.Div(
            [
                dbc.Button("previous", id="btn-prev-image", color="secondary"),
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
    if view_mode == TIME_SERIES:
        return timeseries_layout
    elif view_mode == EVALUATION:
        return timeseries_layout


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port="8504")
