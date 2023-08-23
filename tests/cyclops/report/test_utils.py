"""Report utils tests."""
import json
import os
import shutil
from datetime import date as dt_date

import pytest

from cyclops.report.utils import (
    extract_performance_metrics,
    filter_results,
    flatten_results_dict,
    get_metrics_trends,
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
    assert "2023-07-07: 10:00:00" in metrics_dict  # pylint: disable=C0201
    assert "2023-08-07: 11:00:00" in metrics_dict  # pylint: disable=C0201
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
    assert "2023-07-07: 10:00:00" in metrics_dict  # pylint: disable=C0201
    assert "2023-08-07: 11:00:00" in metrics_dict  # pylint: disable=C0201
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
