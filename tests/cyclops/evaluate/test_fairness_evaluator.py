"""Integration tests for cyclops.evaluate.fairness.evaluator.evaluate_fairness().

evaluate_fairness() (989 lines) previously had zero test coverage despite
being the module's main entry point for fairness/subgroup evaluation.
"""

import pytest
from datasets import Dataset

from cyclops.evaluate.fairness.evaluator import evaluate_fairness
from cyclops.evaluate.metrics.experimental import BinaryAccuracy
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict


@pytest.fixture
def classification_dataset() -> Dataset:
    """Create a small synthetic binary classification dataset."""
    data = {
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "prediction": [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        "group": ["A"] * 6 + ["B"] * 6,
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    }
    return Dataset.from_dict(data)


def test_evaluate_fairness_basic(classification_dataset):
    """Basic categorical group fairness evaluation."""
    metrics = MetricDict([BinaryAccuracy()])
    results = evaluate_fairness(
        metrics=metrics,
        dataset=classification_dataset,
        groups="group",
        target_columns="target",
        prediction_columns="prediction",
    )

    assert set(results.keys()) == {"group:A", "group:B", "overall"}
    for slice_result in results.values():
        assert "BinaryAccuracy" in slice_result
        assert "BinaryAccuracy Parity" in slice_result
        assert 0 <= float(slice_result["BinaryAccuracy"]) <= 1

    # parity relative to the overall metric value must be 1.0 for "overall" itself
    assert float(results["overall"]["BinaryAccuracy Parity"]) == pytest.approx(1.0)


def test_evaluate_fairness_group_base_values(classification_dataset):
    """Parity must be computed relative to an explicit group_base_values."""
    metrics = MetricDict([BinaryAccuracy()])
    results = evaluate_fairness(
        metrics=metrics,
        dataset=classification_dataset,
        groups="group",
        target_columns="target",
        prediction_columns="prediction",
        group_base_values={"group": "A"},
    )

    accuracy_a = float(results["group:A"]["BinaryAccuracy"])
    parity_a = float(results["group:A"]["BinaryAccuracy Parity"])
    # base group's parity relative to itself must be 1.0
    assert parity_a == pytest.approx(1.0)
    assert accuracy_a > 0


def test_evaluate_fairness_group_bins_continuous(classification_dataset):
    """Continuous groups must be bucketed via group_bins."""
    metrics = MetricDict([BinaryAccuracy()])
    results = evaluate_fairness(
        metrics=metrics,
        dataset=classification_dataset,
        groups="age",
        target_columns="target",
        prediction_columns="prediction",
        group_bins={"age": 3},
    )

    assert "overall" in results
    # binning into 3 groups should yield multiple non-overall slice keys
    assert len(results) > 2
    for slice_result in results.values():
        if slice_result["sample_size"] > 0:
            assert "BinaryAccuracy" in slice_result


def test_evaluate_fairness_invalid_dataset_type():
    """A non-Dataset `dataset` argument must raise TypeError."""
    metrics = MetricDict([BinaryAccuracy()])
    with pytest.raises(TypeError, match="Dataset"):
        evaluate_fairness(
            metrics=metrics,
            dataset="not a dataset",  # type: ignore[arg-type]
            groups="group",
            target_columns="target",
            prediction_columns="prediction",
        )


def test_evaluate_fairness_missing_group_column_raises(classification_dataset):
    """A missing group column must raise ValueError."""
    metrics = MetricDict([BinaryAccuracy()])
    with pytest.raises(ValueError, match="missing_group"):
        evaluate_fairness(
            metrics=metrics,
            dataset=classification_dataset,
            groups="missing_group",
            target_columns="target",
            prediction_columns="prediction",
        )


def test_evaluate_fairness_invalid_array_lib(classification_dataset):
    """An unsupported array_lib must raise NotImplementedError."""
    metrics = MetricDict([BinaryAccuracy()])
    with pytest.raises(NotImplementedError):
        evaluate_fairness(
            metrics=metrics,
            dataset=classification_dataset,
            groups="group",
            target_columns="target",
            prediction_columns="prediction",
            array_lib="not_a_real_lib",  # type: ignore[arg-type]
        )
