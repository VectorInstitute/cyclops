from abc import ABC, abstractmethod
import numpy as np
import plotly.graph_objects


# Define an abstract base class SPC (Statistical Process Control)
class BaseSPC(ABC):
    @abstractmethod
    def fit(self, data):
        """
        Abstract method to fit the model to the given data.
        """

    @abstractmethod
    def predict(self, data):
        """
        Abstract method to make predictions using the fitted model.
        """

    @abstractmethod
    def plot(self):
        """
        Abstract method to plot the fitted model.
        """

# Define a concrete class ShewhartControlChart that inherits from SPC
class ShewhartControlChart(BaseSPC):
    def __init__(self, warning_level: int, failure_level: int, update: bool = False):
        """
        Initialize a Shewhart Control Chart.

        Parameters
        ----------
        warning_level : int
            The number of standard deviations from the mean to use as the warning level.
        failure_level : int
            The number of standard deviations from the mean to use as the failure level.
        update : bool, optional
            Whether to update the control limits with new data, by default False.
        """

        self.warning_level = warning_level
        self.failure_level = failure_level
        self.update = update

        self.lcls = {"warning": None, "failure": None}
        self.ucls = {"warning": None, "failure": None}
        self.historical_data = None
        self.out_of_control_points = {"warning": None, "failure": None}


    def fit(self, data: np.ndarray) -> None:
        """
        Implementation of fit method for Shewhart Control Chart.

        Use the data to calculate the upper and lower control limits.
        """
        self.historical_data = data
        self.lcls["warning"] = np.mean(data) - self.warning_level * np.std(data)
        self.lcls["failure"] = np.mean(data) - self.failure_level * np.std(data)

        self.ucls["warning"] = np.mean(data) + self.warning_level * np.std(data)
        self.ucls["failure"] = np.mean(data) + self.failure_level * np.std(data)

        self.out_of_control_points["warning"] = np.where(
            (data < self.lcls["warning"])
            | (data > self.ucls["warning"])
        )[0]
        self.out_of_control_points["failure"] = np.where(
            (data < self.lcls["failure"])
            | (data > self.ucls["failure"])
        )[0]

    def predict(self, data: np.ndarray) -> bool:
        """
        Implementation of predict method for Shewhart Control Chart.

        Use the data to predict whether the data is within the control limits.
        """
        # append new data to historical data
        self.historical_data = np.append(self.historical_data, data)

        # update control limits if update is True
        if self.update:
            self.lcls["warning"] = np.mean(self.historical_data) - self.warning_level * np.std(self.historical_data)
            self.lcls["failure"] = np.mean(self.historical_data) - self.failure_level * np.std(self.historical_data)

            self.ucls["warning"] = np.mean(self.historical_data) + self.warning_level * np.std(self.historical_data)
            self.ucls["failure"] = np.mean(self.historical_data) + self.failure_level * np.std(self.historical_data)
        
        warning = np.where(
            (data < self.lcls["warning"]) | (data > self.ucls["warning"])
        )[0]

        failure = np.where(
            (data < self.lcls["failure"]) | (data > self.ucls["failure"])
        )[0]
        self.out_of_control_points["warning"] = np.append(
            self.out_of_control_points["warning"], warning
        )
        self.out_of_control_points["failure"] = np.append(
            self.out_of_control_points["failure"], failure
        )

        return {"warning": warning, "failure": failure}

    def plot(self, xlabels: list = None, metric: str = None) -> plotly.graph_objects.Figure:
        """
        Create a control chart using the data and the fitted model.
        """
        fig = plotly.graph_objects.Figure()
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=xlabels if xlabels else np.arange(len(self.historical_data)),
                y=np.repeat(self.lcls["failure"], len(self.historical_data)),
                mode="lines",
                marker=dict(color="red"),
                line=dict(dash="dash"),
                # fill='tozeroy',  # Apply shading to failure level
                # fillcolor='rgba(255, 0, 0, 0.3)',  # Set red shading color
            )
        )
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=xlabels if xlabels else np.arange(len(self.historical_data)),
                y=np.repeat(self.lcls["warning"], len(self.historical_data)),
                mode="lines",
                marker=dict(color="yellow"),
                line=dict(dash="dash"),
                fill='tonexty',  # Apply shading to warning level
                fillcolor='rgba(255, 255, 0, 0.3)',  # Set yellow shading color
            )
        )
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=xlabels if xlabels else np.arange(len(self.historical_data)),
                y=np.repeat(self.ucls["warning"], len(self.historical_data)),
                mode="lines",
                marker=dict(color="yellow"),
                line=dict(dash="dash"),
                # fill='toself',  # Apply shading to warning level
                # fillcolor='rgba(255, 255, 0, 0.3)',  # Set yellow shading color
            )
        )
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=xlabels if xlabels else np.arange(len(self.historical_data)),
                y=np.repeat(self.ucls["failure"], len(self.historical_data)),
                mode="lines",
                marker=dict(color="red"),
                line=dict(dash="dash"),
                fill='tonexty',  # Apply shading to failure level
                fillcolor='rgba(255, 255, 0, 0.3)',  # Set yellow shading color
            )
        )
        
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=xlabels if xlabels else np.arange(len(self.historical_data)),
                y=self.historical_data,
                mode="markers+lines",
                name=metric if metric else "Metric",
                line=dict(color="blue"),
                marker=dict(color=np.where(
                    (self.historical_data < self.lcls["failure"]) | (self.historical_data > self.ucls["failure"]),
                    "red",
                    "blue"
                )),
            )
        )

        fig.update_layout(
            title="Shewhart Control Chart",
            showlegend=False,
            yaxis=dict(title=metric if metric else "Metric"),
        )
        return fig
