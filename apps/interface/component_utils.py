"""Utility functions for the interface."""
import base64
import io
from datetime import datetime, timezone
from typing import List, Optional

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import html


def flatten_2d_tuple(tuple_of_tuples):
    """Flatten a tuple of tuples."""
    return tuple(element for tupl in tuple_of_tuples for element in tupl)


def generate_table_contents(
    data: pd.DataFrame, display_limit: Optional[int] = None,
) -> List:
    """Generate the table content objects."""
    if display_limit is not None:
        data = data.sample(min([display_limit, len(data)]))

    columns, values = data.columns, data.astype(str).values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    return [html.Thead(header), html.Tbody(rows)]


def table_result(title, id_):
    """Generate table result components."""
    return (
        html.H5(title),
        dmc.Table(striped=True, highlightOnHover=True, id=f"table-{id_}"),
        html.Div(id=f"meta-{id_}"),
        dmc.Space(h=10),
    )


def recently_clicked(clicked_timestamp, seconds=3):
    """Determine whether a button has just been clicked."""
    if clicked_timestamp is None:
        return False

    curr_time = datetime.now(timezone.utc).timestamp() * 1000
    return abs(curr_time - clicked_timestamp) <= seconds * 1000


def multiple_collapse_toggle(n_clicks_timestamps, ids):
    """Toggle multiple collapses given button click times."""
    n_clicks_timestamps = np.array(n_clicks_timestamps, dtype=float)
    try:
        min_id = np.nanargmax(n_clicks_timestamps)
    except ValueError:
        # For the initial callback when none are actually clicked
        return tuple(False for _ in ids)

    # Open only the selected collapse
    collapse_open = [False] * len(ids)
    collapse_open[min_id] = True
    return tuple(collapse_open)


def get_dataframe_info(data):
    """Recreate df.info() as a DataFrame."""
    return pd.DataFrame(
        {
            "Column": data.columns,
            "Non-Null Count": len(data) - data.isnull().sum().values,
            "Dtype": data.dtypes.values,
        },
    )


def upload_to_dataframe(contents):
    """Convert the contents of a CSV upload to a DataFrame."""
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        data = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except ValueError:
        return None, False

    return data, True
