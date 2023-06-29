"""Classification plotter."""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from cyclops.report.plot.base import Plotter
from cyclops.report.plot.utils import bar_plot, create_figure, line_plot, radar_plot


# pylint: disable=use-dict-literal
class ClassificationPlotter(Plotter):
    """Classification plotter."""

    def __init__(
        self,
        task_type: Literal["binary", "multiclass", "multilabel"],
        task_name: Optional[str] = None,
        class_num: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize the plotter.

        Parameters
        ----------
        task_type : Literal['binary', 'multiclass', 'multilabel']
            Classification task type
        task_name : str, optional
            Classification task name, by default None
        class_num : int, optional
            Number of classes or labels, required for multiclass and multilabel tasks, \
            by default None
        class_names : List[str], optional
            Names of the classes or labels, by default None

        """
        super().__init__()
        self.task_name = task_name
        self.task_type = task_type
        self._set_class_num(class_num)  # type: ignore[arg-type]
        self._set_class_names(class_names)  # type: ignore[arg-type]

    def _set_class_num(self, class_num: int) -> None:
        """Set the number of classes or labels.

        Parameters
        ----------
        class_num : int
            Number of classes or labels

        Raises
        ------
        ValueError
            If class_num is not specified for multiclass and multilabel tasks

        """
        if self.task_type in ["multiclass", "multilabel"] and class_num is None:
            raise ValueError(
                "class_num must be specified for multiclass and multilabel tasks"
            )
        if self.task_type == "binary" and class_num is not None:
            assert class_num == 2, "class_num must be 2 for binary tasks"
        self.class_num = class_num if class_num is not None else 2

    def _set_class_names(self, class_names: List[str]) -> None:
        """Set the names of the classes or labels.

        Parameters
        ----------
        class_names : List[str]
            Names of the classes or labels

        """
        if class_names is not None:
            assert (
                len(class_names) == self.class_num
            ), "class_names must be equal to class_num"
        else:
            if self.task_type == "multilabel":
                class_names = [f"Label_{i+1}" for i in range(self.class_num)]
            else:
                class_names = [f"Class_{i+1}" for i in range(self.class_num)]
        self.class_names = class_names

    def roc_curve(
        self,
        roc_curve: Tuple[
            npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
        ],
        auroc: Optional[Union[float, List[float], npt.NDArray[np.float_]]] = None,
        title: Optional[str] = "ROC Curve",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot ROC curve for a single group or subpopulation.

        Parameters
        ----------
        roc_curve : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (fprs, tprs, thresholds)
        auroc : Union[float, list, np.ndarray], optional
            AUROCs, by default None
        title: str, optional
            Plot title, by default "ROC Curve"
        layout : go.Layout, optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatter

        Returns
        -------
        go.Figure
            The figure object.

        """
        fprs = roc_curve[0]
        tprs = roc_curve[1]

        trace = []
        if self.task_type == "binary":
            if auroc is not None:
                assert isinstance(
                    auroc, float
                ), "aurocs must be a float for binary tasks"
                name = f"Model (AUC = {auroc:.2f})"
            else:
                name = "Model"
            trace.append(
                line_plot(
                    x=fprs,
                    y=tprs,
                    trace_name=name,
                    **plot_kwargs,
                )
            )
        else:
            assert (
                len(fprs) == len(tprs) == self.class_num
            ), "fprs and tprs must be of length class_num for \
                multiclass/multilabel tasks"
            for i in range(self.class_num):
                if auroc is not None:
                    assert (
                        len(auroc) == self.class_num  # type: ignore[arg-type]
                    ), "Aurocs must be of length class_num for \
                        multiclass/multilabel tasks"
                    name = f"{self.class_names[i]} (AUC = {auroc[i]:.2f})"  # type: ignore[index] # noqa: E501 # pylint: disable=line-too-long
                else:
                    name = self.class_names[i]
                trace.append(
                    line_plot(
                        x=fprs[i],
                        y=tprs[i],
                        trace_name=name,
                        **plot_kwargs,
                    )
                )

        trace.append(
            line_plot(
                x=[0, 1], y=[0, 1], name="Random Classifier", line=dict(dash="dash")
            )
        )

        xaxis_title = "False Positive Rate"
        yaxis_title = "True Positive Rate"

        fig = create_figure(
            data=trace,
            title=dict(text=title),
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def roc_curve_comparison(
        self,
        roc_curves: Dict[str, Tuple[npt.NDArray[np.float_], ...]],
        aurocs: Optional[
            Dict[str, Union[float, List[float], npt.NDArray[np.float_]]]
        ] = None,
        title: Optional[str] = "ROC Curve Comparison",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot ROC curves for multiple subpopulations or groups.

        Parameters
        ----------
        roc_curves : Dict[str, Tuple]
            Dictionary of roc curves, with keys being the name of the subpopulation
            or group and values being the roc curve tuple (fprs, tprs, thresholds)
        aurocs : Dict[str, Union[float, list, np.ndarray]], optional
            AUROCs for each subpopulation or group specified by name, by default None
        title: str, optional
            Plot title, by default "ROC Curve Comparison"
        layout : Optional[go.Layout], optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatter

        Returns
        -------
        go.Figure
            Figure object.

        """
        trace = []
        if self.task_type == "binary":
            for slice_name, slice_curve in roc_curves.items():
                if aurocs and slice_name in aurocs:
                    assert isinstance(
                        aurocs[slice_name], float
                    ), "Aurocs must be a float for binary tasks"
                    name = f"{slice_name} (AUC = {aurocs[slice_name]:.2f})"
                else:
                    name = slice_name
                fprs = slice_curve[0]
                tprs = slice_curve[1]
                trace.append(
                    line_plot(
                        x=fprs,
                        y=tprs,
                        trace_name=name,
                        **plot_kwargs,
                    )
                )
        else:
            for slice_name, slice_curve in roc_curves.items():
                assert (
                    len(slice_curve[0]) == len(slice_curve[1]) == self.class_num
                ), f"Fprs and tprs must be of length class_num for \
                    multiclass/multilabel tasks in slice {slice_name}"
                for i in range(self.class_num):
                    if aurocs and slice_name in aurocs:
                        assert (
                            len(aurocs[slice_name]) == self.class_num  # type: ignore[arg-type] # noqa: E501 # pylint: disable=line-too-long
                        ), "Aurocs must be of length class_num for \
                            multiclass/multilabel tasks"
                        name = f"{slice_name}, {self.class_names[i]} \
                            (AUC = {aurocs[i]:.2f})"  # type: ignore[index]
                    else:
                        name = f"{slice_name}, {self.class_names[i]}"
                    fprs = slice_curve[0][i]
                    tprs = slice_curve[1][i]
                    trace.append(
                        line_plot(
                            x=fprs,
                            y=tprs,
                            trace_name=name,
                            **plot_kwargs,
                        )
                    )

        trace.append(
            line_plot(
                x=[0, 1], y=[0, 1], name="Random Classifier", line=dict(dash="dash")
            )
        )

        xaxis_title = "False Positive Rate"
        yaxis_title = "True Positive Rate"

        layout_kwargs = {}
        layout_kwargs["title"] = dict(text=title)
        layout_kwargs["xaxis"] = dict(title=xaxis_title)
        layout_kwargs["yaxis"] = dict(title=yaxis_title)

        fig = create_figure(
            data=trace,
            **layout_kwargs,
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def precision_recall_curve(
        self,
        precision_recall_curve: Tuple[
            npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
        ],
        title: Optional[str] = "Precision-Recall Curve",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot precision-recall curve for a single group or subpopulation.

        Parameters
        ----------
        precision_recall_curve : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (recalls, precisions, thresholds)
        title : str, optional
            Plot title, by default "Precision-Recall Curve"
        layout : go.Layout, optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatter

        Returns
        -------
        go.Figure
            The figure object.

        """
        recalls = precision_recall_curve[0]
        precisions = precision_recall_curve[1]

        if self.task_type == "binary":
            trace = line_plot(
                x=recalls,
                y=precisions,
                **plot_kwargs,
            )
        else:
            trace = []
            assert (
                len(recalls) == len(precisions) == self.class_num
            ), "Recalls and precisions must be of length class_num for \
                multiclass/multilabel tasks"
            for i in range(self.class_num):
                trace.append(
                    line_plot(
                        x=recalls[i],
                        y=precisions[i],
                        name=self.class_names[i],
                        **plot_kwargs,
                    )
                )

        xaxis_title = "Recall"
        yaxis_title = "Precision"

        fig = create_figure(
            data=trace,
            title=dict(text=title),
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def precision_recall_curve_comparison(
        self,
        precision_recall_curves: Dict[str, Tuple[npt.NDArray[np.float_], ...]],
        title: Optional[str] = "Precision-Recall Curve Comparison",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot precision-recall curves for multiple groups or subpopulations.

        Parameters
        ----------
        precision_recall_curves : Dict[str, Tuple]
            Dictionary of precision-recall curves, where the key is \
                the group or subpopulation name and the value is a tuple \
                of (recalls, precisions, thresholds)
        layout : Optional[go.Layout], optional
            Customized figure layout, by default None
        title: str, optional
            Plot title, by default "Precision-Recall Curve Comparison"
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatter

        Returns
        -------
        go.Figure
            Figure object

        """
        trace = []
        if self.task_type == "binary":
            for slice_name, slice_curve in precision_recall_curves.items():
                name = f"{slice_name}"
                trace.append(
                    line_plot(
                        x=slice_curve[0],
                        y=slice_curve[1],
                        trace_name=name,
                        **plot_kwargs,
                    )
                )
        else:
            for slice_name, slice_curve in precision_recall_curves.items():
                assert (
                    len(slice_curve[0]) == len(slice_curve[1]) == self.class_num
                ), f"Recalls and precisions must be of length class_num for \
                    multiclass/multilabel tasks in slice {slice_name}"
                for i in range(self.class_num):
                    name = f"{slice_name}: {self.class_names[i]}"
                    trace.append(
                        line_plot(
                            x=slice_curve[0][i],
                            y=slice_curve[1][i],
                            trace_name=name,
                            **plot_kwargs,
                        )
                    )

        xaxis_title = "Recall"
        yaxis_title = "Precision"

        fig = create_figure(
            data=trace,
            title=dict(text=title),
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def metrics_value(
        self,
        metrics: Dict[str, Union[float, List[float], npt.NDArray[Any]]],
        title: Optional[str] = "Metrics",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot values of metrics for a single group or subpopulation.

        This includes metrics such as precision, recall, auroc, and f_beta.

        Parameters
        ----------
        metrics : Dict[str, Union[float, list, np.ndarray]]
            Dictionary of metrics, where the key is the metric name \
            and the value is the metric value
        title : str, optional
            plot title, by default "Metrics"
        layout : Optional[go.Layout], optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Bar
        Returns
        -------
        go.Figure
            Figure object

        """
        if self.task_type == "binary":
            assert all(
                not isinstance(value, (list, np.ndarray)) for value in metrics.values()
            ), ("Metrics must not be of type list or np.ndarray for" "binary tasks")
            trace = bar_plot(
                x=list(metrics.keys()),  # type: ignore[arg-type]
                y=list(metrics.values()),  # type: ignore[arg-type]
                **plot_kwargs,
            )
        else:
            trace = []
            assert all(
                len(value) == self.class_num  # type: ignore[arg-type]
                for value in metrics.values()
            ), "Every metric must be of length class_num for \
                multiclass/multilabel tasks"
            for i in range(self.class_num):
                trace.append(
                    bar_plot(
                        x=list(metrics.keys()),  # type: ignore
                        y=[value[i] for value in metrics.values()],  # type: ignore
                        name=self.class_names[i],
                        **plot_kwargs,
                    )
                )

        xaxis_title = "Metric"
        yaxis_title = "Score"

        fig = create_figure(
            data=trace,
            title=dict(text=title),
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
            barmode="group" if self.task_type in ["multiclass", "multilabel"] else None,
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def metrics_history(
        self,
        metric_history: Dict[str, Union[List[float], npt.NDArray[Any]]],
        time_steps: Optional[List[str]] = None,
        title: Optional[str] = "Metrics History",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot the history of non-curve metrics such as precision, recall, and f_beta \
        for a single group or subpopulation.

        Parameters
        ----------
        metric_history : Dict[str, Union[list, np.ndarray]]
            Dictionary of metric histories, where the key is the metric name \
                and the value is the metric history as a list or np.ndarray
        time_steps : Optional[List[str]], optional
            List of time steps for the metric history used as the x-axis, \
                by default None
        title: str, optional
            Plot title, by default "Metrics History"
        layout : Optional[go.Layout], optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatter

        Returns
        -------
        go.Figure
            Figure object

        """
        trace = []
        if time_steps is not None:
            assert all(
                len(time_steps) == len(value) for value in metric_history.values()
            ), "Time steps must be of the same length as metric values"
        if self.task_type == "binary":
            for metric_name, metric_values in metric_history.items():
                x_values = (
                    time_steps
                    if time_steps is not None
                    else list(range(len(metric_values)))  # type: ignore[arg-type]
                )
                plot = line_plot(
                    x=x_values,  # type: ignore[arg-type]
                    y=metric_values,
                    name=metric_name,
                    **plot_kwargs,
                )
                plot.update(mode="lines+markers")
                trace.append(plot)
        else:
            for metric_name, metric_values in metric_history.items():
                assert all(
                    len(value) == self.class_num for value in metric_values  # type: ignore[arg-type] # noqa: E501 # pylint: disable=line-too-long
                ), "Metric values must be of length class_num for \
                    multiclass/multilabel tasks"
                for i in range(self.class_num):
                    x_values = (
                        time_steps
                        if time_steps is not None
                        else list(range(len(metric_values)))  # type: ignore[arg-type]
                    )
                    name = f"{metric_name}: {self.class_names[i]}"
                    plot = line_plot(
                        x=x_values,  # type: ignore[arg-type]
                        y=metric_values[i],  # type: ignore[arg-type]
                        trace_name=name,
                        **plot_kwargs,
                    )
                    plot.update(mode="lines+markers")
                    trace.append(plot)

        xaxis_title = "Time Step"
        yaxis_title = "Score"

        fig = create_figure(
            data=trace,
            title=dict(text=title),
            xaxis=dict(title=xaxis_title),
            yaxis=dict(title=yaxis_title),
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def metrics_comparison_radar(
        self,
        slice_metrics: Dict[
            str, Dict[str, Union[float, List[float], npt.NDArray[np.float_]]]
        ],
        title: Optional[str] = "Metrics Comparison",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot metrics such as precision, recall, and f_beta for multiple groups or \
        subpopulations in a radar chart.

        Parameters
        ----------
        slice_metrics : Dict[str, Dict[str, Union[float, np.ndarray, list]]]
            Dictionary of metrics, where the key is the slice name and \
                the value is the metric dictionary containing the metric names \
                and values
        title: str, optional
            Plot title, by default "Metrics Comparison"
        layout : Optional[go.Layout], optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatterpolar

        Returns
        -------
        go.Figure
            Figure object

        Raises
        ------
        ValueError
            If the metric values are not of the correct type

        """
        trace = []
        if self.task_type == "binary":
            for slice_name, metrics in slice_metrics.items():
                metric_names = list(metrics.keys())
                assert all(
                    not isinstance(value, (list, np.ndarray))
                    for value in metrics.values()
                ), (
                    "Generic metrics must not be of type list or np.ndarray for"
                    "binary tasks"
                )
                trace.append(
                    radar_plot(
                        radial=list(metrics.values()),  # type: ignore[arg-type]
                        theta=metric_names,  # type: ignore[arg-type]
                        name=slice_name,
                        **plot_kwargs,
                    )
                )
        else:
            for slice_name, metrics in slice_metrics.items():
                metric_names = list(metrics.keys())
                radial_data: List[float] = []
                theta_data: List[float] = []
                for metric_name, metric_values in metrics.items():
                    if isinstance(metric_values, (list, np.ndarray)):
                        assert (
                            len(metric_values) == self.class_num
                        ), "Metric values must be of length class_num for \
                            multiclass/multilabel tasks"
                        radial_data.extend(metric_values)
                        theta = [
                            f"{metric_name}: {self.class_names[i]}"
                            for i in range(self.class_num)
                        ]
                        theta_data.extend(theta)  # type: ignore[arg-type]
                    elif isinstance(metric_values, float):
                        radial_data.append(metric_values)
                        theta_data.append(metric_name)  # type: ignore[arg-type]
                    else:
                        raise ValueError(
                            "Metric values must be either a number or \
                            of length class_num for multiclass/multilabel tasks"
                        )
                trace.append(
                    radar_plot(
                        radial=radial_data,
                        theta=theta_data,
                        name=slice_name,
                        **plot_kwargs,
                    )
                )

        fig = create_figure(
            data=trace,
            title=dict(text=title),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
        )
        if layout is not None:
            fig.update_layout(layout)
        return fig

    def metrics_comparison_bar(
        self,
        slice_metrics: Dict[
            str, Dict[str, Union[float, List[float], npt.NDArray[np.float_]]]
        ],
        title: Optional[str] = "Metrics Comparison",
        layout: Optional[go.Layout] = None,
        **plot_kwargs: Any,
    ) -> go.Figure:
        """Plot values of metrics for multiple group or subpopulation.

        This includes metrics such as precision, recall, auroc, and f_beta.

        Parameters
        ----------
        slice_metrics : Dict[str, Dict[str, Union[float, np.ndarray, list]]]
            Dictionary of metrics, where the key is the slice name and \
                the value is the metric dictionary containing the metric names \
                and values
        title: str, optional
            Plot title, by default "Metrics Comparison"
        layout : Optional[go.Layout], optional
            Customized figure layout, by default None
        **plot_kwargs : dict
            Additional keyword arguments for plotly.graph_objects.Scatterpolar

        Returns
        -------
        go.Figure
            Figure object

        Raises
        ------
        ValueError
            If the metric values are not of the correct type

        """
        trace = []
        if self.task_type == "binary":
            for slice_name, metrics in slice_metrics.items():
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                assert all(
                    not isinstance(value, (list, np.ndarray))
                    for value in metrics.values()
                ), (
                    "Generic metrics must not be of type list or np.ndarray for"
                    "binary tasks"
                )
                trace.append(
                    bar_plot(
                        x=metric_names,  # type: ignore[arg-type]
                        y=metric_values,  # type: ignore[arg-type]
                        name=slice_name,
                        **plot_kwargs,
                    )
                )

            xaxis_title = "Metric"
            yaxis_title = "Score"

            fig = create_figure(
                data=trace,
                title=dict(text=title),
                xaxis=dict(title=xaxis_title),
                yaxis=dict(title=yaxis_title),
                barmode="group",
            )

        else:
            rows = len(slice_metrics)
            fig = make_subplots(
                rows=rows,
                cols=1,
                subplot_titles=list(slice_metrics.keys()),
                x_title="Metric",
                y_title="Score",
            )

            if len(self.template.layout.colorway) >= self.class_num:
                colors = self.template.layout.colorway[: self.class_num]
            else:
                difference = self.class_num - len(self.template.layout.colorway)
                colors = (
                    self.template.layout.colorway
                    + self.template.layout.colorway[:difference]
                )

            for i, (slice_name, metrics) in enumerate(slice_metrics.items()):
                metric_names = list(metrics.keys())
                for num in range(self.class_num):
                    for metric_name in metric_names:
                        if isinstance(metrics[metric_name], (list, np.ndarray)):
                            metric_values = metrics[metric_name][num]  # type: ignore
                        else:
                            metric_values = metrics[metric_name]  # type: ignore
                    fig.append_trace(
                        bar_plot(
                            x=metric_names,  # type: ignore[arg-type]
                            y=metric_values,  # type: ignore[arg-type]
                            name=self.class_names[num],
                            legendgroup=self.class_names[num],
                            showlegend=(i == 0),
                            marker_color=colors[num],
                            **plot_kwargs,
                        ),
                        row=i + 1,
                        col=1,
                    )

            xaxis_title = "Metric"
            yaxis_title = "Score"
            fig.update_layout(
                title=dict(text=title), barmode="group", height=rows * 300
            )

        if layout is not None:
            fig.update_layout(layout)
        return fig
