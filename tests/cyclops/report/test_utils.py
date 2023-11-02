"""Report utils tests."""


import pytest

from cyclops.report.utils import (
    flatten_results_dict,
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
