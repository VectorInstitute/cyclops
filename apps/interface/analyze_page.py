"""Analyze page components."""
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from component_utils import table_result
from consts import APP_PAGE_ANALYZE
from dash import dcc, html

upload_components = (
    html.H2("Analyze"),
    html.Label("Specify filename from results"),
    dmc.Space(h=10),
    dcc.Input(
        id="server-filename",
        type="text",
        debounce=True,
        placeholder="File name",
        style={"width": "30%"},
    ),
    dbc.Button(
        "Upload",
        id="server-upload-button",
        className="mb-3",
        color="primary",
        n_clicks=0,
        style={"width": 200},
    ),
    dmc.Space(h=20),
    html.Label("OR Select local files"),
    dcc.Upload(
        id="local-upload",
        children=html.Div(["Drag and Drop or Click to ", html.A("Select Files")]),
        style={
            "width": "30%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        multiple=False,
    ),
    dmc.Space(h=10),
    html.Div(id=f"{APP_PAGE_ANALYZE}-loading-message"),
    dmc.Space(h=10),
)

data_option_components = (
    html.H3("Options"),
    dmc.Space(h=10),
    html.Label("Row display limit"),
    dmc.Space(h=5),
    dcc.Input(
        5,
        id=f"{APP_PAGE_ANALYZE}-display-limit",
        type="number",
        min=1,
        max=50,
        style={"width": 100},
    ),
    dcc.Loading(
        children=[html.Div([html.Div(id=f"{APP_PAGE_ANALYZE}-loading-output")])],
        type="circle",
    ),
    dmc.Space(h=30),
)

data_info_components = (
    html.H3("Data Analysis"),
    *table_result("Info", f"{APP_PAGE_ANALYZE}-info"),
    dmc.Space(h=20),
    *table_result("Preview", f"{APP_PAGE_ANALYZE}-preview"),
)

column_info_components = (
    html.H3("Column Analysis"),
    html.Label("Column name"),
    dmc.Space(h=5),
    dcc.Input(
        id=f"{APP_PAGE_ANALYZE}-column-input",
        type="text",
        style={"width": 200},
        debounce=True,
    ),
    dmc.Space(h=20),
    *table_result("Unique value counts (top 50)", f"{APP_PAGE_ANALYZE}-unique-values"),
    dmc.Space(h=10),
)

column_visualization_components = (
    html.H3("Column Visualization"),
    html.Label("Select plots"),
    dcc.Dropdown(
        ["Montreal", "San Francisco"],
        multi=True,
    ),
    dmc.Space(h=10),
)

analyze_page_components = (
    *upload_components,
    dmc.Space(h=20),
    *data_option_components,
    dmc.Space(h=20),
    *data_info_components,
    dmc.Space(h=20),
    *column_info_components,
    dmc.Space(h=20),
    *column_visualization_components,
)
