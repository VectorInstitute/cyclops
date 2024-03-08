"""Test MetricCollection class."""

import inspect

import pytest

from cyclops.evaluate.metrics import MetricCollection
from cyclops.evaluate.metrics.metric import _METRIC_REGISTRY

from .conftest import NUM_CLASSES, NUM_LABELS
from .helpers import _assert_allclose
from .inputs import _binary_cases, _multiclass_cases, _multilabel_cases


@pytest.fixture(name="binary_metrics")
def fixture_binary_metrics():
    """Return binary metrics that require no parameters."""
    return [
        v()
        for k, v in _METRIC_REGISTRY.items()
        if k.startswith("binary_")
        and all(
            p.default is not p.empty for p in inspect.signature(v).parameters.values()
        )
    ]


@pytest.fixture(name="multiclass_metrics")
def fixture_multiclass_metrics():
    """Return multiclass metrics that require no parameters except `num_classes."""
    return [
        v(num_classes=NUM_CLASSES)
        for k, v in _METRIC_REGISTRY.items()
        if k.startswith("multiclass_")
        and all(
            p.default is not p.empty
            for p_name, p in inspect.signature(v).parameters.items()
            if p_name != "num_classes"
        )
    ]


@pytest.fixture(name="multilabel_metrics")
def fixture_multilabel_metrics():
    """Return multilabel metrics that require no parameters except `num_labels`."""
    return [
        v(num_labels=NUM_LABELS)
        for k, v in _METRIC_REGISTRY.items()
        if k.startswith("multilabel_")
        and all(
            p.default is not p.empty
            for p_name, p in inspect.signature(v).parameters.items()
            if p_name != "num_labels"
        )
    ]


def _run_metric_collection_test(metrics, target, preds) -> None:
    """Run metric collection test."""
    assert preds.shape[0] == target.shape[0]
    num_batches = preds.shape[0]

    metric_dict = {m.__class__.__name__: m for m in metrics}

    metric_collection = MetricCollection(metrics)

    for i in range(num_batches):
        metric_dict_batch_results = {}
        for metric_name, metric in metric_dict.items():
            metric_dict_batch_results[metric_name] = metric(target[i], preds[i])

        metric_collection_batch_results = metric_collection(target[i], preds[i])

        _assert_allclose(
            metric_dict_batch_results,
            metric_collection_batch_results,
            atol=1e-8,
        )

    metric_dict_global_results = {}
    for metric_name, metric in metric_dict.items():
        metric_dict_global_results[metric_name] = metric.compute()

    metric_collection_global_results = metric_collection.compute()

    _assert_allclose(
        metric_dict_global_results,
        metric_collection_global_results,
        atol=1e-8,
    )


@pytest.mark.parametrize("inputs", _binary_cases[2:])  # exclude integer case
@pytest.mark.filterwarnings("ignore::UserWarning")  # ignore divide by zero warning
def test_binary_metric_collection(inputs, binary_metrics) -> None:
    """Test binary metric collection."""
    target, preds = inputs
    _run_metric_collection_test(binary_metrics, target, preds)


@pytest.mark.parametrize("inputs", _multiclass_cases[2:])  # exclude integer case
@pytest.mark.filterwarnings("ignore::UserWarning")  # ignore divide by zero warning
def test_multiclass_metric_collection(inputs, multiclass_metrics) -> None:
    """Test multilabel metric collection."""
    target, preds = inputs
    _run_metric_collection_test(multiclass_metrics, target, preds)


@pytest.mark.parametrize("inputs", _multilabel_cases[2:])  # exclude integer case
@pytest.mark.filterwarnings("ignore::UserWarning")  # ignore divide by zero warning
def test_multilabel_metric_collection(inputs, multilabel_metrics) -> None:
    """Test mulitlabel metric collection."""
    target, preds = inputs
    _run_metric_collection_test(multilabel_metrics, target, preds)
