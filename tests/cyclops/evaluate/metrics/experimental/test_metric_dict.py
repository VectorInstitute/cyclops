"""Test the `MetricDict` class."""
from copy import deepcopy

import numpy as np
import numpy.array_api as anp
import pytest
import torch

from cyclops.evaluate.metrics.experimental import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassPrecision,
    MulticlassRecall,
)
from cyclops.evaluate.metrics.experimental.f_score import MulticlassF1Score
from cyclops.evaluate.metrics.experimental.metric_dict import MetricDict

from ..conftest import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES
from .testers import DummyListStateMetric, DummyMetric


def test_empty_metric_dict():
    """Test that an empty `MetricDict` is created when no arguments are passed."""
    metrics = MetricDict()
    assert len(metrics) == 0


@pytest.mark.parametrize(
    "metrics, args, kwargs",
    [
        (None, (), {}),
        (DummyMetric(), (), {}),
        (BinaryAccuracy(), (), {}),
        ((DummyMetric(), DummyListStateMetric()), (), {}),
        ({"metric_a": DummyMetric(), "metric_b": DummyListStateMetric()}, (), {}),
        (None, (DummyMetric(), DummyListStateMetric()), {}),
        (None, (), {"metric_a": DummyMetric(), "metric_b": DummyListStateMetric()}),
        (DummyMetric(), (DummyListStateMetric(),), {"accuracy": BinaryAccuracy()}),
        ({"accuracy": BinaryAccuracy()}, (), {"confmat": BinaryConfusionMatrix()}),
    ],
)
def test_metric_dict_init(metrics, args, kwargs):
    """Test that a `MetricDict` is created with the correct keys."""
    metric_d = MetricDict(metrics, *args, **kwargs)
    num_metrics = (
        (
            len(metrics)
            if isinstance(metrics, (dict, tuple))
            else (1 if metrics is not None else 0)
        )
        + len(args)
        + len(kwargs)
    )
    all_keys = {}
    if isinstance(metrics, dict):
        all_keys.update(metrics)
    elif isinstance(metrics, tuple):
        all_keys.update({metric.__class__.__name__: metric for metric in metrics})
    elif metrics:
        all_keys[metrics.__class__.__name__] = metrics
    if args:
        all_keys.update({metric.__class__.__name__: metric for metric in args})
    if kwargs:
        all_keys.update(kwargs)
    print(metric_d, all_keys)
    assert len(metric_d) == num_metrics
    assert all(key in metric_d for key in all_keys)


@pytest.mark.parametrize(
    ("prefix", "postfix"),
    [
        (None, None),
        ("prefix_", None),
        (None, "_postfix"),
        ("prefix_", "_postfix"),
    ],
)
def test_metric_dict_adfix(prefix, postfix):
    """Test that the `MetricDict` can be created with a prefix and/or postfix."""
    metrics = MetricDict(
        DummyMetric(),
        DummyListStateMetric(),
        prefix=prefix,
        postfix=postfix,
    )
    names = ["DummyMetric", "DummyListStateMetric"]
    names = [f"{prefix}{name}" if prefix else name for name in names]
    names = [f"{name}{postfix}" if postfix else name for name in names]

    # test __call__
    output = metrics(anp.asarray(1, dtype=anp.float32))
    for name in names:
        assert (
            name in output
        ), f"`MetricDict` output does not contain metric {name} when called."

    # test `compute`
    output = metrics.compute()
    for name in names:
        assert (
            name in output
        ), f"`MetricDict` output does not contain metric {name} using the `compute` method."

    # test `clone`
    new_metrics = metrics.clone(prefix="new_")
    output = new_metrics(anp.asarray(1, dtype=anp.float32))
    names = [  # remove old prefix
        n[len(prefix) :] if prefix is not None else n for n in names
    ]
    for name in names:
        assert (
            f"new_{name}" in output
        ), f"`MetricDict` output does not contain metric new_{name} when cloned."

    for k in new_metrics:
        assert "new_" in k

    for k in new_metrics.keys(keep_base=False):
        assert "new_" in k

    for k in new_metrics.keys(keep_base=True):
        assert "new_" not in k

    new_metrics = new_metrics.clone(postfix="_new")
    output = new_metrics(anp.asarray(1, dtype=anp.float32))
    names = [
        n[: -len(postfix)] if postfix is not None else n for n in names
    ]  # remove old postfix
    for name in names:
        assert f"new_{name}_new" in output, (
            f"`MetricDict` output does not contain metric new_{name}_new "
            f"when cloned with prefix and postfix."
        )


def test_invalid_inputs():
    """Test that an error is raised when invalid inputs are passed."""
    with pytest.raises(
        TypeError,
        match="The argument `other_metrics` can only be used if `metrics` is a "
        "single metric or a sequence of metrics.",
    ):
        MetricDict({"accuracy": BinaryAccuracy()}, DummyMetric())

    with pytest.raises(
        TypeError,
        match="Expected `metrics` to be a sequence containing at least one "
        "metric object, but got either an empty sequence or a sequence.*",
    ):
        MetricDict((5, 4, "a"))
    with pytest.raises(
        TypeError,
        match="Expected `metrics` to be a dictionary mapping metric names to "
        "metric objects, but got an empty dictionary or a dictionary.*",
    ):
        MetricDict({"a": 5, "b": 4})
    with pytest.raises(
        TypeError,
        match="Expected `other_metrics` to be a sequence containing at least one "
        "metric object, but got either an empty sequence or a sequence.*",
    ):
        MetricDict(DummyMetric(), 5, 4, 3)
    with pytest.raises(
        TypeError,
        match="Expected `kwargs` to contain at least one metric object, but found "
        "only non-metric objects.*",
    ):
        MetricDict(DummyMetric(), a=5, b=4)
    with pytest.raises(
        TypeError,
        match="Expected `prefix` to be a string, but got int.*",
    ):
        MetricDict(DummyMetric(), prefix=5)
    with pytest.raises(
        TypeError,
        match="Expected `postfix` to be a string, but got int.*",
    ):
        MetricDict(DummyMetric(), postfix=5)

    with pytest.warns(
        UserWarning,
        match="Found object in `metrics` that is not `Metric` or `TorchMetric`. "
        "This object will be ignored: not_a_metric",
    ):
        MetricDict((DummyMetric(), "not_a_metric"))
    with pytest.warns(
        UserWarning,
        match="Found object in `other_metrics` that is not `Metric` or `TorchMetric`. "
        "This object will be ignored: not_a_metric",
    ):
        MetricDict(DummyMetric(), DummyListStateMetric(), "not_a_metric")
    with pytest.warns(
        UserWarning,
        match="Found object in `kwargs` that is not `Metric` or `TorchMetric`. "
        "This object will be ignored: 5",
    ):
        MetricDict(dummy=DummyMetric(), not_a_metric=5)


def test_metric_dict_computation():
    """Test that using `MetricDict` works the same as using the individual metrics."""
    metric1 = DummyMetric()
    metric2 = DummyListStateMetric()
    metrics = MetricDict(metric1, metric2)

    assert len(metrics) == 2
    assert metrics["DummyMetric"] is metric1
    assert metrics["DummyListStateMetric"] is metric2

    # test `update`
    metrics.update(anp.asarray(1, dtype=anp.float32))
    assert metrics["DummyMetric"].x == anp.asarray(1, dtype=anp.float32)
    assert metrics["DummyListStateMetric"].x == [anp.asarray(1, dtype=anp.float32)]

    # test `compute`
    output = metrics.compute()
    assert len(output) == 2
    assert output["DummyMetric"] == anp.asarray(1, dtype=anp.float32)
    assert output["DummyListStateMetric"] == [anp.asarray(1, dtype=anp.float32)]

    # test `reset`
    metrics.reset()
    assert metrics["DummyMetric"].x == anp.asarray(0, dtype=anp.float32)
    assert metrics["DummyListStateMetric"].x == []


_mc_inputs = (
    anp.asarray(np.random.randint(0, NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))),
    anp.asarray(np.random.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
)


@pytest.mark.parametrize(
    "metrics, expected_groups, inputs",
    [
        # single metric forms its own group
        (
            MulticlassAccuracy(num_classes=NUM_CLASSES),
            {0: ["MulticlassAccuracy"]},
            _mc_inputs,
        ),
        # two metrics of same class forms a group
        (
            {
                "acc0": MulticlassAccuracy(num_classes=NUM_CLASSES),
                "acc1": MulticlassAccuracy(num_classes=NUM_CLASSES),
            },
            {0: ["acc0", "acc1"]},
            _mc_inputs,
        ),
        # two metrics with the same state form a group
        (
            [
                MulticlassPrecision(num_classes=NUM_CLASSES),
                MulticlassRecall(num_classes=NUM_CLASSES),
            ],
            {0: ["MulticlassPrecision", "MulticlassRecall"]},
            _mc_inputs,
        ),
        # two metrics with different states form different groups
        (
            [
                MulticlassConfusionMatrix(num_classes=NUM_CLASSES),
                MulticlassRecall(num_classes=NUM_CLASSES),
            ],
            {0: ["MulticlassConfusionMatrix"], 1: ["MulticlassRecall"]},
            _mc_inputs,
        ),
        # multi group multi metric
        (
            [
                MulticlassConfusionMatrix(num_classes=NUM_CLASSES),
                MulticlassPrecision(num_classes=NUM_CLASSES),
                MulticlassRecall(num_classes=NUM_CLASSES),
            ],
            {
                0: ["MulticlassConfusionMatrix"],
                1: ["MulticlassPrecision", "MulticlassRecall"],
            },
            _mc_inputs,
        ),
        # Complex example
        (
            {
                "acc": MulticlassAccuracy(num_classes=NUM_CLASSES),
                "acc2": MulticlassAccuracy(num_classes=NUM_CLASSES),
                "acc3": MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro"),
                "f1": MulticlassF1Score(num_classes=NUM_CLASSES),
                "recall": MulticlassRecall(num_classes=NUM_CLASSES),
                "confmat": MulticlassConfusionMatrix(num_classes=NUM_CLASSES),
            },
            {0: ["acc", "acc2", "f1", "recall"], 1: ["acc3"], 2: ["confmat"]},
            _mc_inputs,
        ),
        # TODO: add list states
    ],
)
@pytest.mark.parametrize(
    ("prefix", "postfix"),
    [
        (None, None),
        ("prefix_", None),
        (None, "_postfix"),
        ("prefix_", "_postfix"),
    ],
)
@pytest.mark.parametrize("with_reset", [True, False])
def test_metric_grouping(metrics, expected_groups, inputs, prefix, postfix, with_reset):
    """Test that metrics are grouped correctly."""
    metrics = MetricDict(deepcopy(metrics), prefix=prefix, postfix=postfix)
    assert not hasattr(metrics, "_metric_groups")

    target, preds = inputs
    bsz = target.shape[0]

    for i in range(bsz):
        metrics.update(target=target[i, ...], preds=preds[i, ...])

        for member in metrics.values():
            assert member._update_count == 1 if with_reset else i + 1

        assert metrics._metric_groups == expected_groups

        metrics.compute()

        if with_reset:
            metrics.reset()


@pytest.mark.integration_test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_to_device():
    """Test that `to_device` works correctly."""
    metrics = MetricDict(DummyMetric(), DummyListStateMetric())
    metrics.to_device("cuda")

    metrics.update(torch.tensor(42, device="cuda"))  # type: ignore

    assert metrics["DummyMetric"].x.device.type == "cuda"
    assert metrics["DummyListStateMetric"].x[0].device.type == "cuda"

    metrics.to_device("cpu")
    assert metrics["DummyMetric"].x.device.type == "cpu"
    assert metrics["DummyListStateMetric"].x[0].device.type == "cpu"
