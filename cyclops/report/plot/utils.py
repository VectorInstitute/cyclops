"""Utility functions for plotting."""
import base64
from typing import List, Union

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.basedatatypes import BaseTraceType


def image_to_base64(
    image_path: str,
) -> str:
    """Convert image to base64 string.

    Parameters
    ----------
    image_path : str
        path to the image file

    Returns
    -------
    str
        The base64 string of the image

    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def fig_to_html(
    fig: go.Figure,
    include_plotlyjs: bool = False,
    full_html: bool = False,
    **kwargs,
) -> str:
    """Convert figure to html.

    Parameters
    ----------
    fig : go.Figure
        Figure object
    include_plotlyjs : bool, optional
        Whether to include plotly.js in the returned html, by default False
    full_html : bool, optional
        Whether to include the full html document, by default False

    Returns
    -------
    str
        HTML string

    """
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=full_html, **kwargs)


def fig_to_image(  # pylint: disable=redefined-builtin
    fig: go.Figure,
    format: str = "png",
    scale: int = 1,
    **kwargs,
) -> bytes:
    """Get image bytes from figure.

    Parameters
    ----------
    fig : go.Figure
        Figure object
    format : str, optional
        Image format, by default 'png'
    scale : int, optional
        Image scale factor, by default 1

    Returns
    -------
    bytes
        Image bytes

    """
    return fig.to_image(format=format, scale=scale, **kwargs)


def save_fig(fig: go.Figure, path: str) -> None:
    """Save figure to path.

    Parameters
    ----------
    fig : go.Figure
        Figure object
    path : str
        Path to save figure to

    """
    pio.write_image(fig, path)


def line_plot(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    trace_name: str = None,
    **kwargs,
) -> go.Scatter:
    """Create a line plot.

    Parameters
    ----------
    x : Union[list, np.ndarray]
        X-axis values
    y : Union[list, np.ndarray]
        y-axis values
    trace_name : str, optional
        Name of the trace, by default None

    Returns
    -------
    go.Scatter
        The line plot

    """
    trace = go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=trace_name,
    )
    trace.update(**kwargs)
    return trace


def radar_plot(  # pylint: disable=invalid-name
    r: Union[list, np.ndarray],
    theta: Union[list, np.ndarray],
    trace_name: str = None,
    **kwargs,
) -> go.Scatterpolar:
    """Create a radar plot.

    Parameters
    ----------
    r : Union[list, np.ndarray]
        radial values
    theta : Union[list, np.ndarray]
        theta values
    trace_name : str, optional
        Name of the trace, by default None

    Returns
    -------
    go.Scatterpolar
        The radar plot

    """
    trace = go.Scatterpolar(
        r=r,
        theta=theta,
        name=trace_name,
        fill="toself",
    )
    trace.update(**kwargs)
    return trace


def bar_plot(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    trace_name: str = None,
    **kwargs,
) -> go.Bar:
    """Create a bar plot.

    Parameters
    ----------
    x : Union[list, np.ndarray]
        X-axis values
    y : Union[list, np.ndarray]
        y-axis values
    trace_name : str, optional
        Name of the trace, by default None

    Returns
    -------
    go.Bar
        The bar plot

    """
    trace = go.Bar(
        x=x,
        y=y,
        name=trace_name,
    )
    trace.update(**kwargs)
    return trace


def create_figure(
    data: Union[BaseTraceType, List[BaseTraceType]],
    **kwargs,
) -> go.Figure:
    """Create a figure.

    Parameters
    ----------
    data : Union[BaseTraceType, List[BaseTraceType]]
        The traces to plot

    Returns
    -------
    go.Figure
        The figure

    """
    fig = go.Figure(
        data=data,
    )
    layout = go.Layout(**kwargs)
    fig.update_layout(layout)
    return fig
