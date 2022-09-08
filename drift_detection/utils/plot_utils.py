import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC curve (area = %0.6f)" % roc_auc)
    return ax


def plot_pr(ax, recall, precision, roc_prc):
    ax.step(recall, precision, color="b", alpha=0.2, where="post")
    ax.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.axis(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.05)
    ax.set_title("PRC curve (area = %0.6f)" % roc_prc)
    return ax


def setup_plot(
    plot_handle: mpl.axes.SubplotBase,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: list,
):
    """Setup plot.

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


def set_bars_color(bars: mpl.container.BarContainer, color: str):
    """Set color attribute for bars in bar plots.

    Parameters
    ----------
    bars: mpl.container.BarContainer
        Bars.
    color: str
        Color.
    """
    for bar in bars:
        bar.set_color(color)


def plot_label_distribution(X, y, label, features):
    data = pd.concat([X, y], axis=1)
    data_pos = data.loc[data[label] == 1]
    data_neg = data.loc[data[label] == 0]
    fig, axs = plt.subplots(2, 2, figsize=(30, 15), tight_layout=True)

    # Across age.
    ages = data[AGE]
    ages_pos = data_pos[AGE]
    ages_neg = data_neg[AGE]
    print(
        f"Mean Age: Outcome present: {np.array(ages_pos).mean()}, No outcome: {np.array(ages_neg).mean()}"
    )

    (_, bins, _) = axs[0][0].hist(ages, bins=50, alpha=0.5, color="g")
    axs[0][0].hist(ages_pos, bins=bins, alpha=0.5, color="r")
    setup_plot(
        axs[0][0],
        "Age distribution",
        "Age",
        "Num. of encounters",
        ["All", "Outcome present"],
    )

    # Across sex.
    sex = list(data[SEX].unique())
    sex_counts = list(data[SEX].value_counts())
    sex_counts_pos = list(data_pos[SEX].value_counts())

    sex_bars = axs[0][1].bar(sex, sex_counts, alpha=0.5)
    set_bars_color(sex_bars, "g")
    sex_bars_pos = axs[0][1].bar(sex, sex_counts_pos, alpha=0.5)
    set_bars_color(sex_bars_pos, "r")
    setup_plot(
        axs[0][1],
        "Sex distribution",
        "Sex",
        "Num. of encounters",
        ["All", "Outcome present"],
    )

    # Across features.
    n = len(features)
    w = 0.04
    x = np.arange(0, len([0, 1]))
    # NUM_COLORS = 20
    # cm = plt.get_cmap('gist_rainbow')

    for i, feature in enumerate(features):
        feature_counts = list(data[feature].value_counts())
        feature_counts_pos = list(data_pos[feature].value_counts())
        if len(feature_counts) == 1:
            feature_counts.append(0)
        if len(icd_counts_pos) == 1:
            feature_counts_pos.append(0)
        position = x + (w * (1 - n) / 2) + i * w
        feature_bars = axs[1][0].bar(position, feature_counts, width=w, alpha=0.5)
        set_bars_color(feature_bars, "g")
        feature_bars_pos = axs[1][0].bar(
            position, feature_counts_pos, width=w, alpha=0.5
        )
        set_bars_color(feature_bars_pos, "r")

    setup_plot(
        axs[1][0],
        "Feature distribution",
        "Feature",
        "Num. of encounters",
        ["All", "Outcome present"],
    )

    # Across labels.
    label_counts = y.value_counts().to_dict().values()
    labels = data[label].value_counts().to_dict().keys()

    label_bars = axs[1][1].bar(labels, label_counts, alpha=0.5)
    set_bars_color(label_bars, "g")
    setup_plot(
        axs[1][1], "Outcome distribution", "Outcome", "Num. of encounters", ["All"]
    )

    plt.show()
