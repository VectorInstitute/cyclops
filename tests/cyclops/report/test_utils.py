"""Report utils tests."""

import json
import os
import shutil
from datetime import date as dt_date

import pytest

from cyclops.report.model_card import ModelCard
from cyclops.report.model_card.fields import (
    Graphic,
    MetricCard,
    PerformanceMetric,
    Test,
)
from cyclops.report.model_card.sections import (
    GraphicsCollection,
    MetricCardCollection,
    Overview,
    QuantitativeAnalysis,
)
from cyclops.report.utils import (
    create_metric_card_plot,
    create_metric_cards,
    extract_performance_metrics,
    filter_results,
    flatten_results_dict,
    get_histories,
    get_metrics_trends,
    get_names,
    get_passed,
    get_slices,
    get_thresholds,
    get_timestamps,
    sweep_graphics,
    sweep_metric_cards,
    sweep_metrics,
    sweep_tests,
)


def test_flatten_results_dict():
    """Test flatten_results_dict function."""
    results = {
        "model1": {
            "slice1": {"metric1": 0.1, "metric2": 0.2},
            "slice2": {"metric1": 0.3, "metric2": 0.4},
        },
        "model2": {
            "slice1": {"metric1": 0.5, "metric2": 0.6},
            "slice2": {"metric1": 0.7, "metric2": 0.8},
        },
    }

    expected = {
        "model1": {
            "slice1/metric1": 0.1,
            "slice1/metric2": 0.2,
            "slice2/metric1": 0.3,
            "slice2/metric2": 0.4,
        },
        "model2": {
            "slice1/metric1": 0.5,
            "slice1/metric2": 0.6,
            "slice2/metric1": 0.7,
            "slice2/metric2": 0.8,
        },
    }
    actual = flatten_results_dict(results)
    assert actual == expected

    remove_metrics = ["metric1"]
    remove_slices = ["slice2"]
    model_name = "model1"
    expected = {
        "slice1/metric2": 0.2,
    }
    actual = flatten_results_dict(
        results,
        remove_metrics=remove_metrics,
        remove_slices=remove_slices,
        model_name=model_name,
    )
    assert actual == expected

    model_name = "model3"
    with pytest.raises(AssertionError):
        flatten_results_dict(results, model_name=model_name)


def test_filter_results():
    """Test filter_results function."""
    results = [
        {"type": "metric1", "value": 10, "slice": "slice1"},
        {"type": "metric2", "value": 20, "slice": "slice1"},
        {"type": "metric1", "value": 30, "slice": "slice2"},
        {"type": "metric2", "value": 40, "slice": "slice2"},
    ]
    expected = [
        {"type": "metric1", "value": 10, "slice": "slice1"},
        {"type": "metric1", "value": 30, "slice": "slice2"},
    ]
    actual = filter_results(results, metric_names="metric1")
    assert actual == expected

    expected = [{"type": "metric1", "value": 30, "slice": "slice2"}]
    actual = filter_results(results, metric_names="metric1", slice_names="slice2")
    assert actual == expected


def test_extract_performance_metrics():
    """Test extract_performance_metrics function."""
    tmpdir = "test_report"
    os.makedirs(os.path.join(tmpdir, "2023-07-07", "10:00:00"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "2023-08-07", "11:00:00"), exist_ok=True)

    model_card1 = {
        "quantitative_analysis": {
            "performance_metrics": [
                {"type": "accuracy", "value": 0.85, "slice": "overall"},
                {"type": "precision", "value": 0.8, "slice": "overall"},
            ],
        },
    }
    model_card2 = {
        "quantitative_analysis": {
            "performance_metrics": [
                {"type": "accuracy", "value": 0.75, "slice": "overall"},
                {"type": "precision", "value": 0.7, "slice": "overall"},
            ],
        },
    }

    with open(
        os.path.join(tmpdir, "2023-07-07", "10:00:00", "model_card.json"),
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(model_card1, json_file)

    with open(
        os.path.join(tmpdir, "2023-08-07", "11:00:00", "model_card.json"),
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(model_card2, json_file)

    metrics_dict = extract_performance_metrics(tmpdir, keep_timestamps=False)
    assert len(metrics_dict) == 2
    assert list(metrics_dict.keys())[0] == "2023-07-07"
    assert (
        metrics_dict["2023-08-07"]
        == model_card2["quantitative_analysis"]["performance_metrics"]
    )

    metrics_dict = extract_performance_metrics(tmpdir, keep_timestamps=True)
    assert len(metrics_dict) == 2
    assert "2023-07-07: 10:00:00" in metrics_dict
    assert "2023-08-07: 11:00:00" in metrics_dict
    assert (
        metrics_dict["2023-07-07: 10:00:00"]
        == model_card1["quantitative_analysis"]["performance_metrics"]
    )
    assert (
        metrics_dict["2023-08-07: 11:00:00"]
        == model_card2["quantitative_analysis"]["performance_metrics"]
    )

    metrics_dict = extract_performance_metrics(
        tmpdir,
        keep_timestamps=True,
        slice_names="overall",
        metric_names="accuracy",
    )
    assert len(metrics_dict) == 2
    assert "2023-07-07: 10:00:00" in metrics_dict
    assert "2023-08-07: 11:00:00" in metrics_dict
    assert metrics_dict["2023-07-07: 10:00:00"] == [
        model_card1["quantitative_analysis"]["performance_metrics"][0],
    ]
    assert metrics_dict["2023-08-07: 11:00:00"] == [
        model_card2["quantitative_analysis"]["performance_metrics"][0],
    ]
    shutil.rmtree(tmpdir)


def test_get_metrics_trends():
    """Test get_metrics_trends function."""
    tmpdir = "test_report"
    os.makedirs(os.path.join(tmpdir, "2023-07-07", "10:00:00"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "2023-08-07", "11:00:00"), exist_ok=True)

    model_card1 = {
        "quantitative_analysis": {
            "performance_metrics": [
                {"type": "accuracy", "value": 0.85, "slice": "overall"},
                {"type": "precision", "value": 0.8, "slice": "overall"},
            ],
        },
    }

    model_card2 = {
        "quantitative_analysis": {
            "performance_metrics": [
                {"type": "accuracy", "value": 0.75, "slice": "overall"},
                {"type": "precision", "value": 0.7, "slice": "overall"},
            ],
        },
    }

    with open(
        os.path.join(tmpdir, "2023-07-07", "10:00:00", "model_card.json"),
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(model_card1, json_file)

    with open(
        os.path.join(tmpdir, "2023-08-07", "11:00:00", "model_card.json"),
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(model_card2, json_file)

    flat_results = {
        "overall/accuracy": 0.5,
        "overall/precision": 0.4,
        "overall/recall": 0.3,
    }

    trends = get_metrics_trends(
        tmpdir,
        flat_results,
        metric_names=["accuracy", "precision"],
        keep_timestamps=False,
    )
    assert len(trends) == 3
    assert list(trends.keys())[0] == "2023-07-07"
    assert len(trends["2023-08-07"]) == 2
    today = dt_date.today().strftime("%Y-%m-%d")
    assert any(
        d["type"] == "accuracy" and d["value"] == 0.85 for d in trends["2023-07-07"]
    )
    assert any(d["type"] == "precision" and d["value"] == 0.4 for d in trends[today])
    shutil.rmtree(tmpdir)


@pytest.fixture(name="model_card")
def model_card():
    """Create a test input for model card."""
    model_card = ModelCard()
    model_card.overview = Overview(
        slices=["overall"],
        metric_cards=MetricCardCollection(
            metrics=["BinaryAccuracy", "BinaryPrecision"],
            slices=["overall"],
            collection=[
                MetricCard(
                    name="Accuracy",
                    type="BinaryAccuracy",
                    slice="overall",
                    tooltip="Accuracy is the proportion of correct predictions among all predictions.",
                    value=0.85,
                    threshold=0.7,
                    passed=True,
                    history=[0.8, 0.85, 0.9],
                    trend="positive",
                    plot=GraphicsCollection(collection=[Graphic(name="Accuracy")]),
                ),
                MetricCard(
                    name="Precision",
                    type="BinaryPrecision",
                    slice="overall",
                    tooltip="Precision is the proportion of correct positive predictions among all positive predictions.",
                    value=0.8,
                    threshold=0.7,
                    passed=True,
                    history=[0.7, 0.8, 0.9],
                    trend="positive",
                    plot=GraphicsCollection(collection=[Graphic(name="Precision")]),
                ),
            ],
        ),
    )
    model_card.quantitative_analysis = QuantitativeAnalysis()
    model_card.quantitative_analysis.performance_metrics = [
        PerformanceMetric(
            type="BinaryAccuracy",
            value=0.85,
            slice="overall",
            tests=[Test()],
        ),
        PerformanceMetric(
            type="BinaryPrecision",
            value=0.8,
            slice="overall",
            tests=[Test()],
        ),
    ]
    return model_card


def test_sweep_tests(model_card):
    """Test sweep_tests function."""
    tests = []
    sweep_tests(model_card, tests)
    assert len(tests) == 2


def test_sweep_metrics(model_card):
    """Test sweep_metrics function."""
    metrics = []
    sweep_metrics(model_card, metrics)
    assert len(metrics) == 2


def test_sweep_metric_cards(model_card):
    """Test sweep_metric_cards function."""
    metric_cards = []
    sweep_metric_cards(model_card, metric_cards)
    assert len(metric_cards) == 2


def test_sweep_graphics(model_card):
    """Test sweep_graphics function."""
    graphics = []
    sweep_graphics(model_card, graphics, caption="Precision")
    assert len(graphics) == 1


def test_get_slices(model_card):
    """Test get_slices function."""
    slices = get_slices(model_card)
    # read slices from json to dict
    slices_dict = json.loads(slices)
    assert len(slices_dict.values()) == 2


def test_get_timestamps(model_card):
    """Test get_timestamps function."""
    timestamps = get_timestamps(model_card)
    # read timestamps from json to dict
    timestamps_dict = json.loads(timestamps)
    assert len(timestamps_dict.values()) == 2


def test_get_histories(model_card):
    """Test get_plots function."""
    plots = get_histories(model_card)
    # read plots from json to dict
    plots_dict = json.loads(plots)
    assert len(plots_dict.values()) == 2


def test_get_thresholds(model_card):
    """Test get_thresholds function."""
    thresholds = get_thresholds(model_card)
    # read thresholds from json to dict
    thresholds_dict = json.loads(thresholds)
    assert len(thresholds_dict.values()) == 2


def test_get_passed(model_card):
    """Test get_passed function."""
    passed = get_passed(model_card)
    # read passed from json to dict
    passed_dict = json.loads(passed)
    assert len(passed_dict.values()) == 2


def test_get_names(model_card):
    """Test get_names function."""
    names = get_names(model_card)
    # read names from json to dict
    names_dict = json.loads(names)
    assert len(names_dict.values()) == 2


def test_create_metric_cards(model_card):
    """Test create_metric_cards function."""
    timestamp = "2021-01-01"
    current_metrics = []
    sweep_metrics(model_card, metrics=current_metrics)
    metric_cards = create_metric_cards(
        current_metrics=current_metrics[0],
        timestamp=timestamp,
    )[-1]
    assert len(metric_cards) == 2
    current_metrics = [
        PerformanceMetric(
            type="BinaryAccuracy",
            value=0.85,
            slice="overall",
            description="Accuracy of binary classification",
            graphics=None,
            tests=None,
        ),
        PerformanceMetric(
            type="MulticlassPrecision",
            value=[0.9, 0.8, 0.7],
            slice="class:0",
            description="Precision of multiclass classification",
            graphics=None,
            tests=None,
        ),
    ]
    timestamp = "2022-01-01"
    last_metric_cards = [
        MetricCard(
            name="BinaryAccuracy",
            type="BinaryAccuracy",
            slice="overall",
            tooltip="Accuracy of binary classification",
            value=0.8,
            threshold=0.9,
            passed=False,
            history=[0.75, 0.8, 0.85],
            timestamps=["2021-01-01", "2021-02-01", "2021-03-01"],
        ),
        MetricCard(
            name="MulticlassPrecision",
            type="MulticlassPrecision",
            slice="class:0",
            tooltip="Precision of multiclass classification",
            value=0.85,
            threshold=0.9,
            passed=True,
            history=[0.8, 0.85, 0.9],
            timestamps=["2021-01-01", "2021-02-01", "2021-03-01"],
        ),
    ]
    metrics, tooltips, slices, values, metric_cards = create_metric_cards(
        current_metrics=current_metrics,
        timestamp=timestamp,
        last_metric_cards=last_metric_cards,
    )
    assert metrics == ["Accuracy", "Precision"]
    assert tooltips == [
        "Accuracy of binary classification",
        "Precision of multiclass classification",
    ]
    assert slices == ["class"]
    assert values == [["0"]]
    assert len(metric_cards) == 2
    assert isinstance(metric_cards[0], MetricCard)
    assert isinstance(metric_cards[1], MetricCard)
    assert metric_cards[0].name == "Accuracy"
    assert metric_cards[1].name == "Precision"


def test_create_metric_card_plot():
    """Test create_metric_card_plot function."""
    metric_card_plot = create_metric_card_plot(history=[0.7, 0.8, 0.9], threshold=0.7)
    assert isinstance(metric_card_plot, GraphicsCollection)
