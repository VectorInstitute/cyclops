"""Integration tests for the top-level evaluate() function.

These exercise cyclops.evaluate.evaluator.evaluate() end-to-end against a
small in-memory dataset - previously the main public entry point for
evaluating models had zero test coverage.
"""

import pytest
from datasets import Dataset, DatasetDict
from datasets.splits import Split

from cyclops.data.slicer import SliceSpec
from cyclops.evaluate.evaluator import evaluate
from cyclops.evaluate.fairness.config import FairnessConfig
from cyclops.evaluate.metrics.experimental import BinaryAccuracy, BinaryPrecision
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict


@pytest.fixture
def classification_dataset() -> Dataset:
    """Create a small synthetic binary classification dataset."""
    data = {
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "prediction": [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
        "group": ["A"] * 6 + ["B"] * 6,
    }
    return Dataset.from_dict(data)


def test_evaluate_basic(classification_dataset):
    """evaluate() with a single metric and no slicing computes an overall result."""
    metrics = MetricDict([BinaryAccuracy()])
    results = evaluate(
        dataset=classification_dataset,
        metrics=metrics,
        target_columns="target",
        prediction_columns="prediction",
    )

    assert "model_for_prediction" in results
    overall = results["model_for_prediction"]["overall"]
    assert 0 <= float(overall["BinaryAccuracy"]) <= 1
    assert overall["sample_size"] == classification_dataset.num_rows


def test_evaluate_with_slice_spec(classification_dataset):
    """evaluate() with a slice_spec computes per-slice results."""
    metrics = MetricDict([BinaryAccuracy()])
    slice_spec = SliceSpec(
        spec_list=[{"group": {"value": "A"}}, {"group": {"value": "B"}}],
    )

    results = evaluate(
        dataset=classification_dataset,
        metrics=metrics,
        target_columns="target",
        prediction_columns="prediction",
        slice_spec=slice_spec,
    )

    model_results = results["model_for_prediction"]
    assert set(model_results.keys()) == {"group:A", "group:B", "overall"}
    assert model_results["group:A"]["sample_size"] == 6
    assert model_results["group:B"]["sample_size"] == 6
    assert model_results["overall"]["sample_size"] == 12


def test_evaluate_multiple_prediction_columns(classification_dataset):
    """evaluate() with multiple prediction columns computes results per model."""
    dataset = classification_dataset.add_column(
        "prediction_2",
        [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    )
    metrics = MetricDict([BinaryAccuracy()])
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        target_columns="target",
        prediction_columns=["prediction", "prediction_2"],
    )

    assert set(results.keys()) == {"model_for_prediction", "model_for_prediction_2"}


def test_evaluate_empty_slice_raises(classification_dataset):
    """An empty slice must raise when raise_on_empty_slice=True."""
    metrics = MetricDict([BinaryAccuracy()])
    slice_spec = SliceSpec(
        spec_list=[{"group": {"value": "nonexistent"}}],
        include_overall=False,
    )
    with pytest.raises(RuntimeError, match="empty"):
        evaluate(
            dataset=classification_dataset,
            metrics=metrics,
            target_columns="target",
            prediction_columns="prediction",
            slice_spec=slice_spec,
            raise_on_empty_slice=True,
        )


def test_evaluate_empty_slice_warns_and_returns_nan(classification_dataset):
    """An empty slice must warn and produce NaN metric values by default."""
    metrics = MetricDict([BinaryAccuracy()])
    slice_spec = SliceSpec(
        spec_list=[{"group": {"value": "nonexistent"}}],
        include_overall=False,
    )
    with pytest.warns(RuntimeWarning, match="empty"):
        results = evaluate(
            dataset=classification_dataset,
            metrics=metrics,
            target_columns="target",
            prediction_columns="prediction",
            slice_spec=slice_spec,
            raise_on_empty_slice=False,
        )

    slice_result = results["model_for_prediction"]["group:nonexistent"]
    assert slice_result["BinaryAccuracy"] != slice_result["BinaryAccuracy"]  # NaN


def test_evaluate_missing_required_column_raises(classification_dataset):
    """A missing target/prediction column must raise ValueError."""
    metrics = MetricDict([BinaryAccuracy()])
    with pytest.raises(ValueError, match="missing_column"):
        evaluate(
            dataset=classification_dataset,
            metrics=metrics,
            target_columns="missing_column",
            prediction_columns="prediction",
        )


def test_evaluate_dataset_dict_without_split_uses_choose_split(
    classification_dataset,
):
    """A DatasetDict with split=None must fall back to choose_split(), not error."""
    dataset_dict = DatasetDict({"test": classification_dataset})
    metrics = MetricDict([BinaryAccuracy()])
    results = evaluate(
        dataset=dataset_dict,
        metrics=metrics,
        target_columns="target",
        prediction_columns="prediction",
    )
    assert "model_for_prediction" in results


def test_evaluate_dataset_dict_split_all_raises(classification_dataset):
    """A DatasetDict with split=Split.ALL must raise ValueError."""
    dataset_dict = DatasetDict({"test": classification_dataset})
    metrics = MetricDict([BinaryAccuracy()])
    with pytest.raises(ValueError, match="Split.ALL"):
        evaluate(
            dataset=dataset_dict,
            metrics=metrics,
            target_columns="target",
            prediction_columns="prediction",
            split=Split.ALL,
        )


def test_evaluate_with_fairness_config(classification_dataset):
    """evaluate() with a fairness_config populates a "fairness" results key."""
    metrics = MetricDict([BinaryAccuracy(), BinaryPrecision()])
    fairness_config = FairnessConfig(
        metrics=metrics,
        dataset=classification_dataset,  # overridden by evaluate() with the real dataset
        groups="group",
        target_columns="target",
    )

    results = evaluate(
        dataset=classification_dataset,
        metrics=metrics,
        target_columns="target",
        prediction_columns="prediction",
        fairness_config=fairness_config,
    )

    assert "fairness" in results
    fairness_results = results["fairness"]
    assert set(fairness_results.keys()) == {"group:A", "group:B", "overall"}
    for group_result in fairness_results.values():
        assert "BinaryAccuracy" in group_result
        assert "BinaryAccuracy Parity" in group_result
