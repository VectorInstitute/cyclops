"""Test classification plotter in cyclops report."""

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from cyclops.report.plot.classification import ClassificationPlotter, PRCurve, ROCCurve


def _rand_input():
    """
    Generate a random array of 10 numbers.

    Returns
    -------
      A NumPy array of shape (10,) containing random floats between 0 and 1.
    """
    return np.random.rand(10)


def _multiclass_rand_input():
    """
    Generate a random array of 10 ndarrays.

    Returns
    -------
      A NumPy array of shape (3, 10) containing random floats between 0 and 1.
      We have 10 samples and 3 classes.
    """
    return np.random.rand(3, 10)


_binary_plotter = ClassificationPlotter(
    task_type="binary",
    task_name="b_plot",
    class_num=2,
    class_names=["#1", "#2"],
)
_mclass_plotter = ClassificationPlotter(
    task_type="multiclass",
    task_name="mclass_plot",
    class_num=3,
    class_names=["#1", "#2", "#3"],
)
_mlabel_plotter = ClassificationPlotter(
    task_type="multilabel",
    task_name="mlabel_plot",
    class_num=3,
    class_names=["#1", "#2", "#3"],
)

_slices = [f"slice_#{i}" for i in range(5)]
_metrics = ["recall", "f_beta"]


def binary_slice_metrics():
    """Generate slice metrics dictionary for binary classification."""
    input_dic = defaultdict(lambda: defaultdict(float))
    for i in range(len(_slices)):
        for metric in _metrics:
            input_dic[_slices[i]][metric] = np.random.rand()

    return input_dic


def multiclass_slice_metrics():
    """Generate slice metrics dictionary for multi-class classification."""
    input_dic = defaultdict(lambda: defaultdict(list))
    for i in range(len(_slices)):
        for metric in _metrics:
            input_dic[_slices[i]][metric] = np.random.rand(3)
    return input_dic


def test_plot_threshperf():
    """Test the threshperf plot."""
    fpr = np.array([0.1, 0.2, 0.3])
    tpr = np.array([0.8, 0.9, 0.95])
    thresholds = np.array([0.4, 0.6, 0.8])
    ppv = np.array([0.7, 0.8, 0.9])
    npv = np.array([0.6, 0.7, 0.8])
    pred_probs = np.array([0.5, 0.6, 0.7])

    roc_curve = ROCCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)

    plot = _binary_plotter.threshperf(roc_curve, ppv, npv, pred_probs)
    assert isinstance(plot, go.Figure)


def test_plot_calibration():
    """Test the calibration plot."""
    data = pd.DataFrame(
        {
            "y_true": np.random.randint(0, 2, 100),
            "y_prob": np.random.rand(100),
            "group_col": np.random.randint(0, 2, 100),
        }
    )
    plot = _binary_plotter.calibration(
        data, y_true_col="y_true", y_prob_col="y_prob", group_col="group_col"
    )
    assert isinstance(plot, go.Figure)


def test_plot_roc_curve():
    """Test the ROC plot method for different types of ClassificationPlotters."""
    fpr = np.array([0.1, 0.2, 0.3])
    tpr = np.array([0.8, 0.9, 0.95])
    thresholds = np.array([0.4, 0.6, 0.8])

    roc_curve = ROCCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)
    plot = _binary_plotter.roc_curve(roc_curve)
    assert isinstance(plot, go.Figure)

    fpr, tpr, thresholds = [_multiclass_rand_input() for _ in range(3)]
    roc_curve = ROCCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)
    plot = _mclass_plotter.roc_curve(roc_curve)
    assert isinstance(plot, go.Figure)


def test_plot_precision_recall_curve():
    """Test plotting single-group precision-recall curve."""
    precision = np.array([0.8, 0.7, 0.6])
    recall = np.array([0.9, 0.8, 0.7])
    thresholds = np.array([0.3, 0.5, 0.7])

    pr_curve = PRCurve(precision=precision, recall=recall, thresholds=thresholds)
    plot = _binary_plotter.precision_recall_curve(pr_curve)
    assert isinstance(plot, go.Figure)

    precision, recall, thresholds = [_multiclass_rand_input() for _ in range(3)]
    pr_curve = PRCurve(precision=precision, recall=recall, thresholds=thresholds)

    plot = _mclass_plotter.precision_recall_curve(pr_curve)
    assert isinstance(plot, go.Figure)


def test_roc_curve_comparison():
    """Test plotting multi-group ROC curve."""
    mclass_roc_curves = {}
    bin_roc_curves = {}
    for i in range(3):
        fpr, tpr, thresholds = [_rand_input() for _ in range(3)]
        bin_roc_curves["group_" + str(i)] = ROCCurve(fpr, tpr, thresholds)

        fpr, tpr, thresholds = [_multiclass_rand_input() for _ in range(3)]
        mclass_roc_curves["group_" + str(i)] = ROCCurve(fpr, tpr, thresholds)
    plot = _binary_plotter.roc_curve_comparison(bin_roc_curves)
    assert isinstance(plot, go.Figure)

    plot = _mclass_plotter.roc_curve_comparison(mclass_roc_curves)
    assert isinstance(plot, go.Figure)


def test_precision_recall_curve_comparison():
    """Test plotting multi-group precision-recall curve."""
    mclass_curves = {}
    bin_curves = {}

    for i in range(3):
        prec, rec, thresholds = [_rand_input() for _ in range(3)]
        bin_curves["group_" + str(i)] = PRCurve(prec, rec, thresholds)

        prec, rec, thresholds = [_multiclass_rand_input() for _ in range(3)]
        mclass_curves["group_" + str(i)] = PRCurve(prec, rec, thresholds)

    plot = _binary_plotter.precision_recall_curve_comparison(bin_curves)
    assert isinstance(plot, go.Figure)

    plot = _mclass_plotter.precision_recall_curve_comparison(mclass_curves)
    assert isinstance(plot, go.Figure)


def test_metric_value():
    """Test plotting values of metrics for a single group or subpopulation."""
    metrics = {"f1_score": 0.63, "accuracy": 0.87}
    plot = _binary_plotter.metrics_value(metrics)
    assert isinstance(plot, go.Figure)

    metrics = {"f1_score": [0.64, 0.23, 0.45], "precision": [0.64, 0.23, 0.45]}
    plot = _mclass_plotter.metrics_value(metrics)
    assert isinstance(plot, go.Figure)


def test_metric_trends():
    """Test plotting the trend of non-curve metrics."""
    # date/time (str): [list of dict]
    # each dict = {"type": "f_beta", "value": 1.0, "slice": "#1"} for binary
    # each dict = {"type": "f_beta", "value": [1.0, 2.0], "slice": "#1"} for mclass
    input_dic = defaultdict(list)
    dates = ["Jun", "Jul", "Aug", "Sept", "Oct"]
    for i in range(len(dates)):
        slice_dict = {"type": "f1_score", "value": np.random.rand(), "slice": f"#{i}"}
        input_dic[dates[i]].append(slice_dict)

    plot = _binary_plotter.metrics_trends(input_dic)
    assert isinstance(plot, go.Figure)

    input_dic.clear()
    for i in range(len(dates)):
        slice_dict = {"type": "f1_score", "value": np.random.rand(3), "slice": f"#{i}"}
        input_dic[dates[i]].append(slice_dict)
    print(input_dic)

    plot = _mclass_plotter.metrics_trends(input_dic)
    assert isinstance(plot, go.Figure)


def test_metrics_comparison_radar():
    """Test plotting the trend of non-curve metrics."""
    # input  : {slice name (str): dict}
    # each dict = {"recall": 1.0, "f_beta": 2.0} for binary
    # each dict = {"recall": [1.0, 2.0, 3.0], "f_beta": [1.0, 2.0, 3.0]} for mclass

    input_dic = binary_slice_metrics()
    plot = _binary_plotter.metrics_comparison_radar(input_dic)
    assert isinstance(plot, go.Figure)

    input_dic = multiclass_slice_metrics()
    plot = _mclass_plotter.metrics_comparison_radar(input_dic)
    assert isinstance(plot, go.Figure)


# @pytest.mark.skip(reason="Probably there's a bug that needs fixing")
def test_metrics_comparison_bar():
    """Test plotting the trend of non-curve metrics."""
    plot = _binary_plotter.metrics_comparison_bar(binary_slice_metrics())
    assert isinstance(plot, go.Figure)
    input_dict = multiclass_slice_metrics()
    import pprint

    pprint.pprint(input_dict)
    print(input_dict)
    plot = _mclass_plotter.metrics_comparison_bar(input_dict)
    assert isinstance(plot, go.Figure)


def test_metrics_comparison_scatter():
    """Test plotting the trend of non-curve metrics."""
    plot = _binary_plotter.metrics_comparison_scatter(binary_slice_metrics())
    assert isinstance(plot, go.Figure)
    input_dict = multiclass_slice_metrics()
    import pprint

    pprint.pprint(input_dict)
    print(input_dict)
    plot = _mclass_plotter.metrics_comparison_scatter(input_dict)
    assert isinstance(plot, go.Figure)


def test_confusion_matrix():
    """Test plotting confusion matrix."""
    bin_conf_mat = np.random.rand(2, 2)
    plot = _binary_plotter.confusion_matrix(bin_conf_mat)
    assert isinstance(plot, go.Figure)

    mclass_conf_mat = np.random.rand(3, 3)
    plot = _mclass_plotter.confusion_matrix(mclass_conf_mat)
    assert isinstance(plot, go.Figure)
