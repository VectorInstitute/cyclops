"""Simple landing page with links to cyclops server UI pages."""


import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from flask_caching import Cache

CACHE_TIMEOUT = 3000
TEXT_ALIGN_CENTER = {"textAlign": "center"}

# Initialize app.
app = dash.Dash(
    external_stylesheets=[dbc.themes.SIMPLEX], suppress_callback_exceptions=True
)
app.title = "cyclops"
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


def create_tiles():
    """Create icon tiles."""
    fig = go.Figure()
    width = 0

    # Add shapes
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0.5,
        y0=-1,
        x1=2.5,
        y1=1,
        line=dict(
            color="LightSeaGreen",
            width=width,
        ),
        fillcolor="LightPink",
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=2.5,
        y0=-1,
        x1=4.5,
        y1=1,
        line=dict(
            color="LightSeaGreen",
            width=width,
        ),
        fillcolor="PaleTurquoise",
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=2.5,
        y0=1,
        x1=4.5,
        y1=3,
        line=dict(
            color="LightSeaGreen",
            width=width,
        ),
        fillcolor="LightSalmon",
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=0.5,
        y0=1,
        x1=2.5,
        y1=3,
        line=dict(
            color="LightSeaGreen",
            width=width,
        ),
        fillcolor="gray",
    )

    # Update axes properties
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_layout(plot_bgcolor="white", height=1000)

    return fig


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Br(),
        html.Div(
            [
                html.Img(src=app.get_asset_url("vector_logo.png"), height=32),
                dcc.Markdown("""# cyclops"""),
            ],
            style=TEXT_ALIGN_CENTER,
        ),
        html.Hr(),
        html.Div(
            dcc.Graph(
                figure=create_tiles(),
                id="tiles",
            ),
            style=TEXT_ALIGN_CENTER,
        ),
    ],
    style=TEXT_ALIGN_CENTER,
)


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port="8132")
