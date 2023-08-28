"""Base class for plotter."""

from typing import Any, Dict, List, Union

import plotly.graph_objects as go
import plotly.io as pio


class Plotter:
    """Plotter base class."""

    def __init__(self) -> None:
        """Initialize plotter."""
        self.set_template("plotly")

    def set_template(
        self,
        template: Union[str, Dict[str, Any], go.layout.Template],
    ) -> None:
        """Set the template for the plotter.

        Parameters
        ----------
        template : Union[str, dict, go.layout.Template]
            The template to use for the plotter

        """
        if isinstance(template, str):
            self.template = pio.templates[template]
        elif isinstance(template, dict):
            self.template = go.layout.Template(template)
        else:
            self.template = template
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"

    def set_colorscale(self, colorscale: str) -> None:
        """Set the colorscale for the plotter.

        Parameters
        ----------
        colorscale : str
            The colorscale to use for the plotter

        """
        self.template.layout.colorscale = colorscale
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"

    def set_colorway(self, colorway: List[str]) -> None:
        """Set the colorway for the plotter.

        Parameters
        ----------
        colorway : list
            The colorway to use for the plotter

        """
        self.template.layout.colorway = colorway
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"

    def set_background_color(self, color: str) -> None:
        """Set the background color for the plotter.

        Parameters
        ----------
        color : str
            The background color to use for the plotter

        """
        self.template.layout.plot_bgcolor = color
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"

    def set_paper_color(self, color: str) -> None:
        """Set the page color for the plotter.

        Parameters
        ----------
        color : str
            The background color to use for the plotter

        """
        self.template.layout.paper_bgcolor = color
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"

    def set_grid(self, grid: bool) -> None:
        """Set the grid for the plotter.

        Parameters
        ----------
        grid : bool
            Whether to show grid for the plotter

        """
        self.template.layout.xaxis.showgrid = grid
        self.template.layout.yaxis.showgrid = grid
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"

    def set_font(self, font: Dict[str, Any]) -> None:
        """Set the font for the plotter.

        Parameters
        ----------
        font : dict
            The font to use for the plotter with keys 'family', 'size', and 'color'.

        """
        self.template.layout.font = font
        pio.templates["new_template"] = self.template
        pio.templates.default = "new_template"
