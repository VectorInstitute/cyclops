import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shift_constants import *

# Plotting parametersai
linestyles = ["-", "-.", "--", ":"]
format = ["-o", "-h", "-p", "-s", "-D", "-<", "->", "-X"]
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


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip("#")

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, fmt="-o", label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogx(x, y, fmt, color=color, label=label)
    ax.fill_between(
        x, np.clip(ymax, 0, 1), np.clip(ymin, 0, 1), color=color, alpha=alpha_fill
    )


def plot_roc(ax, fpr, tpr, roc_auc):
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--")
    ax.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve (area = %0.6f)" % roc_auc)
    return ax


def plot_pr(ax, recall, precision, average_precision):
    ax.step(recall, precision, color="b", alpha=0.2, where="post")
    ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
    ax.set_title("Average Precision: {0:0.6f}".format(average_precision))
    return ax
