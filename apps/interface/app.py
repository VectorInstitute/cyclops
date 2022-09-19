"""Main interface application file."""

from datetime import datetime
from os import path
from typing import Any, Dict, List, Optional, Tuple

import app_query
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from component_utils import (
    generate_table_contents,
    get_dataframe_info,
    recently_clicked,
    upload_to_dataframe,
)
from consts import (
    APP_DIAG,
    APP_ENC,
    APP_PAGE_ANALYZE,
    APP_PAGE_QUERY,
    CACHE_TIMEOUT,
    NAV_PAGES,
    TABLE_IDS,
)
from css import TEXT_ALIGN_CENTER
from dash import Dash, Input, Output, State, dcc, dependencies, html
from flask_caching import Cache
from tabs.analyze_tab import analyze_page_components
from tabs.query_tab import query_page_components
from tabs.visualizer_tab import encounters_events, events, visualizer_page_components

from cyclops.plotter import plot_timeline
from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.utils.file import join, load_dataframe, save_dataframe

ANALYZE_DATA = None

app = Dash(external_stylesheets=[dbc.themes.UNITED], suppress_callback_exceptions=True)
app.title = "cyclops.interface"
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


# APP LAYOUT
# ------------------------------------------------------------------------------
tabs_components = html.Div(
    [
        dmc.Tabs(
            id="nav-tabs",
            active=0,
            color="grape",
            children=[dmc.Tab(label=f"{page}") for _, page in enumerate(NAV_PAGES)],
        ),
        html.Div(id="tab-content"),
    ]
)


# App layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(src=app.get_asset_url("vector_logo.png"), height=32),
                dcc.Markdown(
                    """# CyclOps Interface""",
                    style={"background-color": "grape"},
                ),
            ],
            style=TEXT_ALIGN_CENTER,
        ),
        html.Hr(),
        tabs_components,
    ],
    style={"padding-left": "20px", "padding-top": "20px"},
)

# QUERY PAGE FUNCTIONALITY
# ------------------------------------------------------------------------------


@app.callback(Output("tab-content", "children"), Input("nav-tabs", "active"))
def tab_selection(active):
    """Navbar tab selection."""
    if active == 0:
        return query_page_components
    if active == 1:
        return analyze_page_components

    return visualizer_page_components


# Encounters checkbox / collapse
@app.callback(
    Output(f"{APP_ENC}-collapse", "is_open"),
    [Input(f"{APP_ENC}-checkbox", "checked")],
)
def toggle_collapse_encounter(checked):
    """Toggle encounter collapse."""
    return checked


# Diagnoses checkbox / collapse
@app.callback(
    Output(f"{APP_DIAG}-collapse", "is_open"),
    [Input(f"{APP_DIAG}-checkbox", "checked")],
)
def toggle_collapse_diagnosis(checked):
    """Toggle diagnosis collapse."""
    return checked


# Advanced options collapse/expand
@app.callback(
    Output("advanced-collapse", "is_open"),
    [Input("advanced-collapse-button", "n_clicks")],
    [State("advanced-collapse", "is_open")],
)
def toggle_advanced_options(n_clicks, is_open):
    """Toggle diagnosis collapse with a button."""
    if n_clicks is None or n_clicks == 0:
        return False

    return not is_open


# Run the query
@app.callback(
    [
        Output(f"{APP_PAGE_QUERY}-loading-output", "children"),
        *(Output(f"table-{id_}-preview", "children") for id_ in TABLE_IDS),
        *(Output(f"meta-{id_}-preview", "children") for id_ in TABLE_IDS),
    ],
    Input("query-run-btn", "n_clicks"),
    [
        State(f"{APP_PAGE_QUERY}-display-limit", "value"),
        State(f"{APP_PAGE_QUERY}-save-checkbox", "checked"),
        State(f"{APP_ENC}-checkbox", "checked"),
        State(f"{APP_ENC}-date-range", "value"),
        State(f"{APP_ENC}-sex", "value"),
        State(f"{APP_ENC}-age-min", "value"),
        State(f"{APP_ENC}-age-max", "value"),
        State(f"{APP_DIAG}-checkbox", "checked"),
        State(f"{APP_DIAG}-code", "value"),
        State(f"{APP_DIAG}-substring", "value"),
    ],
)
def run_query(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    n_clicks: int,
    display_limit: int,
    save_queries_checked: bool,
    encounter_checked: bool,
    date_range: Optional[List[str]],
    sexes: Optional[List[Dict[str, str]]],
    age_min: Optional[float],
    age_max: Optional[float],
    diagnosis_checked: bool,
    diagnosis_codes: Optional[List[Dict[str, Any]]],
    diagnosis_substring: Optional[List[Dict[str, Any]]],
) -> Tuple:
    """Query function."""
    # If the button hasn't been clicked, show no results
    if n_clicks is None or display_limit is None:
        data_outputs = tuple([None] * len(TABLE_IDS))
        data_meta = tuple([""] * len(TABLE_IDS))
    else:
        encounter_kwargs: Dict[str, Any] = {}
        diagnosis_kwargs: Dict[str, Any] = {}

        # Parse patient encounter arguments
        if encounter_checked:
            if date_range is not None:
                encounter_kwargs["after_date"] = str(date_range[0])
                encounter_kwargs["before_date"] = str(date_range[1])

            if sexes is not None:
                encounter_kwargs["sex"] = [d["displayValue"] for d in sexes]

        # Parse diagnosis arguments
        if diagnosis_checked:
            if diagnosis_codes is not None:
                diagnosis_kwargs["diagnosis_codes"] = [
                    d["displayValue"] for d in diagnosis_codes
                ]

            if diagnosis_substring is not None:
                diagnosis_kwargs["diagnosis_substring"] = [
                    d["displayValue"] for d in diagnosis_substring
                ]

        # Query
        datas = app_query.query(
            encounter_checked,
            encounter_kwargs,
            age_min,
            age_max,
            diagnosis_checked,
            diagnosis_kwargs,
        )

        # Save
        if save_queries_checked:
            dt_string = datetime.now().strftime("%Y-%d-%m_%H:%M:%S")
            for key, data in datas.items():
                save_dataframe(data, join("results", dt_string + "_" + key))

        # Prepare metadata
        data_meta = []
        for key in TABLE_IDS:
            if key not in datas:
                data_meta.append("")
            else:
                data = datas[key]
                rows_str = f"{len(data):,}"
                data_meta.append(f"{rows_str} rows  x  {len(data.columns)} columns")

        # Fill non-existing data as empty DataFrames (this works and they don't render)
        for required in TABLE_IDS:
            if required not in datas:
                datas[required] = pd.DataFrame()

        # Prepare the data as table objects
        data_outputs = [
            generate_table_contents(datas[key], display_limit=display_limit)
            for key in TABLE_IDS
        ]

    # Return (loading mechanism , ) + (plotting results) + (plotting result sizes)
    return (None,) + tuple(data_outputs) + tuple(data_meta)


# ANALYZE PAGE FUNCTIONALITY
# ------------------------------------------------------------------------------


@app.callback(
    [
        Output(f"{APP_PAGE_ANALYZE}-loading-message", "children"),
        Output(f"table-{APP_PAGE_ANALYZE}-info", "children"),
        Output(f"table-{APP_PAGE_ANALYZE}-preview", "children"),
    ],
    [
        Input("local-upload", "contents"),
        Input("server-upload-button", "n_clicks_timestamp"),
    ],
    [
        State("server-filename", "value"),
        State(f"{APP_PAGE_ANALYZE}-display-limit", "value"),
    ],
)
def upload_data(
    local_contents, server_upload_click_timestamp, server_filepath, display_limit
):
    """Upload data and display the relevant information."""
    global ANALYZE_DATA  # pylint: disable=global-statement

    if display_limit is None:
        return None, *tuple([None] * 2)

    # Server upload
    if recently_clicked(server_upload_click_timestamp):
        if server_filepath is None or server_filepath == "":
            return None, *tuple([None] * 2)

        server_filepath = join("results", server_filepath)

        if not path.exists(server_filepath):
            error_msg = f"Could not find relative filepath {server_filepath}."
            return error_msg, *tuple([None] * 2)

        data = load_dataframe(server_filepath)

    # Local upload
    else:
        if local_contents is None:
            return None, *tuple([None] * 2)

        data, successful = upload_to_dataframe(local_contents)

        if not successful:
            error_msg = "Could not load local file. Must be a .csv file."
            return error_msg, *tuple([None] * 2)

    data_info = get_dataframe_info(data)
    ANALYZE_DATA = data
    return (
        None,
        generate_table_contents(data_info),
        generate_table_contents(data, display_limit=display_limit),
    )


@app.callback(
    [
        Output(f"table-{APP_PAGE_ANALYZE}-unique-values", "children"),
    ],
    [
        Input(f"{APP_PAGE_ANALYZE}-column-input", "value"),
    ],
)
def analyze_column(col_name):
    """Display relevant information given a column name of the data being analyzed."""
    global ANALYZE_DATA  # pylint: disable=global-statement, W0602

    if ANALYZE_DATA is None:
        return (None,)

    if col_name not in ANALYZE_DATA.columns:
        return (None,)

    # Create value counts table
    value_counts_df = (
        ANALYZE_DATA[col_name].value_counts().iloc[:50].to_frame().reset_index()
    )
    value_counts_df = value_counts_df.rename(
        {"index": "value", col_name: "count"}, axis=1
    )
    value_count_components = generate_table_contents(value_counts_df)
    return (value_count_components,)


# VISUALIZER PAGE FUNCTIONALITY
# ------------------------------------------------------------------------------


@app.callback(
    [
        dependencies.Output("timeline", "figure"),
        dependencies.Output("encounter-events-caption", "children"),
        dependencies.Output("encounter-events-slider", "max"),
    ],
    {
        **{"index": dependencies.Input("encounter-events-slider", "value")},
    },
)
def update_timeline_plot(index):
    """Update timeline plot."""
    events_encounter = events.loc[events[ENCOUNTER_ID] == encounters_events[index]]
    return [
        plot_timeline(events_encounter, return_fig=True),
        f'Encounter ID: "{encounters_events[index]}"',
        len(encounters_events) - 1,
    ]


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port="8504")
