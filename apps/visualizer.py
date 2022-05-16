"""Cyclops Visualizer Application."""


import os
import sys
import logging
import subprocess

import numpy as np
import dash
from dash import html
from dash import dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import flask
from flask_caching import Cache

from css import (
    SIDEBAR_STYLE,
    SIDEBAR_HEADING_STYLE,
    SIDEBAR_LIST_STYLE,
    CONTENT_STYLE,
    TEXT_ALIGN_CENTER,
)
from cyclops.utils.log import setup_logging


CACHE_TIMEOUT = 3000


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
                {"label": "Time-series", "value": SINGLE_IMAGE},
                {"label": "Evaluation", "value": ACTIVE_LEARNING},
            ],
            value=SINGLE_IMAGE,
            labelStyle=SIDEBAR_LIST_STYLE,
        ),
        html.Br(),
        html.Br(),
        html.H3("Data Sources", className="display-6", style=SIDEBAR_HEADING_STYLE),
        html.Br(),
        html.H3("load from file", className="display-6"),
        dcc.Checklist(
            id="tasks_file",
            options=[{"value": task, "label": task} for task in TASKS],
            value=[OBJECT_DETECTION_2D],
            labelStyle=SIDEBAR_LIST_STYLE,
        ),
        html.Br(),
        html.Br(),
    ],
    style=SIDEBAR_STYLE,
)


single_image_layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    id="frame-id",
                    children="",
                ),
                html.Br(),
                html.Img(
                    id="overlay",
                ),
                html.Br(),
                html.Br(),
                dcc.Slider(
                    id="single-image-slider",
                    min=0,
                    max=(num_samples - 1),
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
        html.Div(
            [dcc.Markdown("""# cyclops visualizer""")],
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
                    id="btn-next-image",
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
    if view_mode == SINGLE_IMAGE:
        return single_image_layout
    elif view_mode == GRID:
        return grid_image_layout


if __name__ == "__main__":
    log.info(config)
    app.run_server(debug=True, host="0.0.0.0", port=config.port)

