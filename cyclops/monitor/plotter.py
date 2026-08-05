"""Plotting functions for drift detection."""

from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plotting parameters
linestyles = ["-", "-.", "--", ":"]
lineformat = ["-o", "-h", "-p", "-s", "-D", "-<", "->", "-X"]
markers = ["o", "h", "p", "s", "D", "<", ">", "X"]
brightness = [1.5, 1.25, 1.0, 0.75, 0.5]
colors = [
    "#2196f3",
    "#f44336",
    "#9c27b0",
    "#64dd17",
    "#009688",
    "#ff9800",
    "#795548",
    "#607d8b",
]


def clamp(val: int, minimum: int = 0, maximum: int = 255) -> int:
    """Ensure colour intensity is within a specified range.

    Returns
    -------
    int
        clamped colour intensity.

    """
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr: str, scalefactor: float) -> str:
    """Create color scale.

    Returns
    -------
    String
        Color scale of hex codes.

    """
    hexstr = hexstr.strip("#")

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    red, green, blue = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    red = clamp(int(red * scalefactor))
    green = clamp(int(green * scalefactor))
    blue = clamp(int(blue * scalefactor))

    return f"#{red:02x}{green:02x}{blue:02x}"


def errorfill(
    x: np.ndarray[float, np.dtype[np.float64]],
    y: np.ndarray[float, np.dtype[np.float64]],
    yerr: np.ndarray[float, np.dtype[np.float64]],
    color: Optional[str] = None,
    alpha_fill: float = 0.2,
    ax: Optional[mpl.axes.SubplotBase] = None,
    fmt: str = "-o",
    label: Optional[str] = None,
    semilogx: bool = True,
) -> None:
    """Create custom error fill."""
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    if semilogx:
        ax.semilogx(x, y, fmt, color=color, label=label)
    else:
        ax.plot(x, y, fmt, color=color, label=label)
    ax.fill_between(
        x,
        np.clip(ymax, 0, 1),
        np.clip(ymin, 0, 1),
        color=color,
        alpha=alpha_fill,
    )


def plot_roc(
    ax: mpl.axes.SubplotBase,
    fpr: List[float],
    tpr: List[float],
    roc_auc: str,
) -> mpl.axes.SubplotBase:
    """Plot ROC curve.

    Returns
    -------
    mpl.axes.SubplotBase
        ROC subplot.

    """
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--")
    ax.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC curve (area = {roc_auc:.2f})")
    return ax


def plot_pr(
    ax: mpl.axes.SubplotBase,
    recall: List[float],
    precision: List[float],
    roc_prc: str,
) -> mpl.axes.SubplotBase:
    """Plot Precision-Recall curve.

    Returns
    -------
    mpl.axes.SubplotBase
        PR subplot.

    """
    ax.step(recall, precision, color="b", alpha=0.2, where="post")
    ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
    ax.set_title(f"PRC curve (area = {roc_prc:.2f})")
    return ax


def setup_plot(
    plot_handle: mpl.axes.SubplotBase,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: List[str],
) -> None:
    """Plot setup.

    Parameters
    ----------
    plot_handle: mpl.axes.SubplotBase
        Subplot handle.
    title: str
        Title of plot.
    xlabel: str
        Label for x-axis.
    ylabel: str
        Label for y-axis.
    legend: list
        Legend for different sub-groups.

    """
    plot_handle.title.set_text(title)
    plot_handle.set_xlabel(xlabel, fontsize=20)
    plot_handle.set_ylabel(ylabel, fontsize=20)
    plot_handle.legend(legend, loc=1)


def set_bars_color(bars: mpl.container.BarContainer, color: str) -> None:
    """Set color attribute for bars in bar plots.

    Parameters
    ----------
    bars: mpl.container.BarContainer
        Bars for barplot.
    color: str
        Color for bars in barplot.

    """
    for bar_item in bars:
        bar_item.set_color(color)


def plot_drift_experiment(
    results: dict[str, dict[str, np.ndarray[float, np.dtype[np.float64]]]],
    plot_distance=False,
) -> None:
    """Plot drift experiment p-values.

    Parameters
    ----------
    results: dict
        Dictionary with results from drift experiment.

    """
    if plot_distance:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 16))
        for shift_iter, shift in enumerate(results.keys()):
            samples = results[shift]["samples"]
            mean_p_vals = results[shift]["p_val"].mean(axis=0)
            std_p_vals = results[shift]["p_val"].std(axis=0)
            errorfill(
                samples,
                mean_p_vals,
                std_p_vals,
                fmt=linestyles[shift_iter] + markers[shift_iter],
                color=colorscale(colors[shift_iter], brightness[shift_iter]),
                label=shift,
                ax=ax1,
            )
            ax1.set_xlabel("Number of samples from test")
            ax1.set_ylabel("$p$-value")
            ax1.axhline(y=results[shift]["p_val_threshold"], color="k")
            mean_distance = results[shift]["distance"].mean(axis=0)
            std_distance = results[shift]["distance"].std(axis=0)
            errorfill(
                samples,
                mean_distance,
                std_distance,
                fmt=linestyles[shift_iter] + markers[shift_iter],
                color=colorscale(colors[shift_iter], brightness[shift_iter]),
                label=shift,
                ax=ax2,
            )
            ax2.set_xlabel("Number of samples from test")
            ax2.set_ylabel("Distance")
    else:
        _, ax = plt.subplots(1, 1, figsize=(11, 8))
        for shift_iter, shift in enumerate(results.keys()):
            samples = results[shift]["samples"]
            mean_p_vals = results[shift]["p_val"].mean(axis=0)
            std_p_vals = results[shift]["p_val"].std(axis=0)
            errorfill(
                samples,
                mean_p_vals,
                std_p_vals,
                fmt=linestyles[shift_iter] + markers[shift_iter],
                color=colorscale(colors[shift_iter], brightness[shift_iter]),
                label=shift,
                ax=ax,
            )
            ax.set_xlabel("Number of samples from test")
            ax.set_ylabel("$p$-value")
            ax.axhline(y=results[shift]["p_val_threshold"], color="k")
        plt.legend()
        plt.show()


def plot_drift_timeseries(
    results: dict[str, dict[str, np.ndarray[float, np.dtype[np.float64]]]],
    plot_distance=False,
) -> None:
    """Plot drift results.

    Parameters
    ----------
    results: dict
        Dictionary with results from drift experiment.

    """
    if plot_distance:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        cmap = mpl.colors.ListedColormap(["lightgrey", "red"])
        mean_p_vals = results["p_val"].mean(axis=0)
        std_p_vals = results["p_val"].std(axis=0)
        errorfill(
            results["samples"],
            mean_p_vals,
            std_p_vals,
            fmt=".-",
            color="red",
            label="$p$-value",
            ax=ax1,
            semilogx=False,
        )
        ax1.axhline(
            y=results["p_val_threshold"],
            color="dimgrey",
            linestyle="--",
            label="$p$-value threshold",
        )
        ax1.set_ylabel("$p$-vals", fontsize=16)
        ax1.set_xlabel("timestamp", fontsize=16)
        ax1.set_xticklabels([])
        ax1.pcolorfast(
            ax1.get_xlim(),
            ax1.get_ylim(),
            results["shift_detected"],
            cmap=cmap,
            alpha=0.4,
        )
        ax1.legend()

        mean_distance = results["distance"].mean(axis=0)
        std_distance = results["distance"].std(axis=0)
        errorfill(
            results["samples"],
            mean_distance,
            std_distance,
            fmt=".-",
            color="red",
            label="Distance",
            ax=ax2,
            semilogx=False,
        )
        ax2.set_ylabel("Distance", fontsize=16)
        ax2.set_xlabel("timestamp", fontsize=16)
        ax2.set_xticklabels([])
        ax2.pcolorfast(
            ax2.get_xlim(),
            ax2.get_ylim(),
            results["shift_detected"],
            cmap=cmap,
            alpha=0.4,
        )
        ax2.legend()
        plt.show()
    else:
        _, ax1 = plt.subplots(1, 1, figsize=(16, 10))
        cmap = mpl.colors.ListedColormap(["lightgrey", "red"])
        mean_p_vals = results["p_val"].mean(axis=0)
        std_p_vals = results["p_val"].std(axis=0)
        errorfill(
            results["samples"],
            mean_p_vals,
            std_p_vals,
            fmt=".-",
            color="red",
            label="$p$-value",
            ax=ax1,
            semilogx=False,
        )
        ax1.axhline(y=results["p_val_threshold"], color="dimgrey", linestyle="--")
        ax1.set_ylabel("$p$-vals", fontsize=16)
        ax1.set_xlabel("timestamps", fontsize=16)
        ax1.pcolorfast(
            ax1.get_xlim(),
            ax1.get_ylim(),
            results["shift_detected"],
            cmap=cmap,
            alpha=0.4,
        )
        ax1.legend()
        plt.show()


def plot_performance(results: pd.DataFrame, metric: str) -> None:
    """Plot drift results.

    Parameters
    ----------
    results: pd.DataFrame
        Dataframe returned from rolling window containing dates and performance metric.
    metric: str
        Column name of performance for plotting.

    """
    _, ax1 = plt.subplots(1, 1, figsize=(16, 10))
    ax1.plot(
        results["dates"],
        results[metric],
        ".-",
        color="blue",
        linewidth=0.5,
        markersize=2,
    )
    ax1.set_xlim(results["dates"], results["dates"])
    ax1.set_ylabel(metric, fontsize=16)
    ax1.set_xlabel("time (s)", fontsize=16)
    ax1.axhline(y=np.mean(results[metric]), color="dimgrey", linestyle="--")
    ax1.tick_params(axis="x", labelrotation=45)
    ax1.pcolorfast(
        ax1.get_xlim(),
        ax1.get_ylim(),
        results["detection"][np.newaxis],
        alpha=0.4,
    )

    for index, label in enumerate(ax1.xaxis.get_ticklabels()):
        if index % 28 != 0:
            label.set_visible(False)

    plt.show()
