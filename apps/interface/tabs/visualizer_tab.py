"""Cyclops Visualizer Application."""

import os

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from css import CONTENT_STYLE, SIDEBAR_LIST_STYLE, SIDEBAR_STYLE, TEXT_ALIGN_CENTER
from dash import dcc, html

from cyclops.processors.column_names import ENCOUNTER_ID

STATIC = "static"
TEMPORAL = "temporal"

EVALUATION = "evaluation"
TIMELINE = "timeline"
FEATURE_STORE = "feature_store"
DIRECT_LOAD = "direct_load"

dir_path = os.path.dirname(os.path.abspath(__file__))

events = pd.read_parquet(
    os.path.join(
        "/mnt/data",
        "cyclops/use_cases/mimiciv/mortality_decompensation",
        "data/1-cleaned",
        "batch_0017.parquet",
    )
)
encounters_events = list(events[ENCOUNTER_ID].unique())[0:80]


sidebar_components = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Br(),
        html.H5("Mode"),
        dcc.RadioItems(
            id="view-mode",
            options=[
                {"label": "Visit Timeline", "value": TIMELINE},
                {"label": "Static Features", "value": STATIC},
                {"label": "Temporal Features", "value": TEMPORAL},
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


# timeseries_plot_components = html.Div(
#     [
#         html.Div(
#             [
#                 html.Div(
#                     dcc.Graph(id="temporal-features"),
#                     style={"overflowY": "scroll", "height": 500},
#                 ),
#                 html.Br(),
#                 dcc.Dropdown(
#                     id="temporal-features-dropdown",
#                     value=temporal_features.columns[0],
#                     options=[
#                         {"label": name, "value": name}
#                         for name in temporal_features.columns
#                     ],
#                     multi=True,
#                     searchable=True,
#                 ),
#                 html.Div(id="encounter-caption"),
#                 dcc.Slider(
#                     id="encounter-slider",
#                     min=0,
#                     max=(num_encounters - 1),
#                     step=1,
#                     value=0,
#                     marks=None,
#                     tooltip={"placement": "bottom", "always_visible": True},
#                 ),
#             ]
#         )
#     ],
#     style=TEXT_ALIGN_CENTER,
# )


timeline_plot_components = html.Div(
    [
        html.Div(
            dcc.Graph(id="timeline"),
            style={"overflowY": "scroll", "height": 500},
        ),
        html.Br(),
        html.Div(id="encounter-events-caption"),
        dmc.Slider(
            id="encounter-events-slider",
            step=1,
            value=0,
            persistence=True,
            color="grape",
        ),
    ],
    style=TEXT_ALIGN_CENTER,
)


# static_plot_components = html.Div(
#     [
#         html.Div(
#             [
#                 html.Div(
#                     dcc.Graph(id="static-features"),
#                     style={"overflowY": "scroll", "height": 500},
#                 ),
#                 dcc.Dropdown(
#                     id="static-features-dropdown",
#                     value=static_features.columns[0],
#                     options=[
#                         {"label": name, "value": name}
#                         for name in static_features.columns
#                     ],
#                     multi=False,
#                     searchable=True,
#                 ),
#                 dcc.Slider(
#                     id="encounter-slider",
#                     min=0,
#                     max=(num_encounters - 1),
#                     step=1,
#                     value=0,
#                 ),
#             ]
#         )
#     ],
#     style=TEXT_ALIGN_CENTER,
# )


offcanvas_sidebar_components = html.Div(
    [
        dbc.Button(
            "Options",
            id="options",
            n_clicks=0,
        ),
        dbc.Offcanvas(
            sidebar_components,
            id="offcanvas-scrollable",
            scrollable=True,
            title="Options",
            is_open=False,
        ),
    ]
)


visualizer_page_components = html.Div(
    [
        html.Div(
            [
                html.H2("Visualize"),
            ],
            style={
                "textAlign": "center",
                "background-color": "rgba(214, 212, 208, 0.5)",
            },
        ),
        offcanvas_sidebar_components,
        dcc.Loading(
            id="loading-view",
            type="default",
            color="green",
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
