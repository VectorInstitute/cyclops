"""Query page components."""
import dash_bootstrap_components as dbc
import dash_cool_components as dcool
import dash_mantine_components as dmc
from component_utils import flatten_2d_tuple, table_result
from consts import APP_DIAG, APP_ENC, APP_PAGE_QUERY, TABLE_IDS, TABLES  # , APP_EVENT
from dash import dcc, html

encounter_components = (
    dmc.Checkbox(
        id=f"{APP_ENC}-checkbox",
        label="Patient encounters",
    ),
    dbc.Collapse(
        dbc.Card(
            [
                dmc.Space(h=10),
                dmc.DateRangePicker(
                    id=f"{APP_ENC}-date-range",
                    label="Patient Admission Date Range",
                    style={"width": 330},
                ),
                dmc.Space(h=20),
                html.Label("Sex"),
                dmc.Space(h=50),
                dbc.Card(
                    [
                        dcool.TagInput(
                            id=f"{APP_ENC}-sex",
                            placeholder="Specify sexes (Blank = All)",
                        )
                    ]
                ),
                dmc.Space(h=50),
                html.Label("Age"),
                html.Div(
                    [
                        dcc.Input(
                            id=f"{APP_ENC}-age-min",
                            type="number",
                            placeholder="Min (Blank = No min)",
                            min=0,
                        ),
                        dcc.Input(
                            id=f"{APP_ENC}-age-max",
                            type="number",
                            placeholder="Max (Blank = No max)",
                            min=0,
                        ),
                    ]
                ),
                dmc.Space(h=50),
            ]
        ),
        id=f"{APP_ENC}-collapse",
    ),
    dmc.Space(h=10),
)

diagnosis_components = (
    dmc.Checkbox(
        id=f"{APP_DIAG}-checkbox",
        label="Diagnoses",
    ),
    dbc.Collapse(
        dbc.Card(
            [
                dmc.Space(h=10),
                html.Label("Diagnosis code"),
                dmc.Space(h=50),
                dbc.Card(
                    [
                        dcool.TagInput(
                            id=f"{APP_DIAG}-code",
                            placeholder="Specify codes (Blank = All)",
                        )
                    ]
                ),
                dmc.Space(h=50),
                html.Label("Diagnosis substring"),
                dmc.Space(h=50),
                dbc.Card(
                    [
                        dcool.TagInput(
                            id=f"{APP_DIAG}-substring",
                            placeholder="Specify substrings (Blank = All)",
                        )
                    ]
                ),
                dmc.Space(h=50),
                dmc.Space(h=20),
            ]
        ),
        id=f"{APP_DIAG}-collapse",
    ),
    dmc.Space(h=10),
)

advanced_options_components = (
    dbc.Button(
        "Show Advanced",
        id="advanced-collapse-button",
        className="mb-3",
        color="primary",
        n_clicks=0,
        style={"width": 200},
    ),
    dbc.Collapse(
        dbc.Card(dbc.CardBody("ADVANCED OPTIONS")),
        id="advanced-collapse",
        is_open=False,
    ),
    dmc.Space(h=10),
)

query_option_components = (
    html.H3("Query Options"),
    dmc.Checkbox(
        id=f"{APP_PAGE_QUERY}-save-checkbox",
        label="Save queries",
    ),
    dmc.Space(h=30),
    html.Label("Row display limit"),
    dmc.Space(h=5),
    dcc.Input(
        5,
        id=f"{APP_PAGE_QUERY}-display-limit",
        type="number",
        min=1,
        max=50,
        style={"width": 100},
    ),
    dbc.Button("RUN", id=f"{APP_PAGE_QUERY}-run-btn", style={"width": 100}),
    dcc.Loading(
        children=[html.Div([html.Div(id=f"{APP_PAGE_QUERY}-loading-output")])],
        type="circle",
    ),
    dmc.Space(h=30),
)

query_result_components = (
    html.H3("Results"),
    dmc.Space(h=10),
    *flatten_2d_tuple(
        (*table_result(title, TABLE_IDS[i] + "-preview"), dmc.Space(h=30))
        for i, title in enumerate(TABLES)
    ),
)

query_page_components = (
    html.H2("Cohort Curation"),
    dmc.Space(h=10),
    html.H3("Query"),
    *encounter_components,
    *diagnosis_components,
    dmc.Space(h=40),
    *advanced_options_components,
    dmc.Space(h=30),
    *query_option_components,
    *query_result_components,
)
