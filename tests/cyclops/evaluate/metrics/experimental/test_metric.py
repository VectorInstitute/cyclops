"""Tests for the base class of metrics."""

import array_api_compat as apc
import numpy as np
import numpy.array_api as anp
import pytest
import torch

from cyclops.evaluate.metrics.experimental.utils.ops import (
    dim_zero_cat,
    dim_zero_max,
    dim_zero_mean,
    dim_zero_min,
    dim_zero_sum,
)

from .testers import DummyListStateMetric, DummyMetric


def test_inherit():
    """Test that metric that inherits can be instantiated."""
    DummyMetric()


def test_dist_backend_kwarg():
    """Test different options for `dist_backend` kwarg."""
    with pytest.raises(
        ValueError,
        match="Backend `nonexistent` is not found.*",
    ):
        DummyMetric(dist_backend="nonexistent")

    with pytest.raises(
        TypeError,
        match="Expected `name` to be a str, but got <class 'int'>.",
    ):
        DummyMetric(dist_backend=42)


def test_add_state_factory():
    """Test that `add_state_default_factory` method works as expected."""
    metric = DummyMetric()

    # happy path
    # default_factory is callable with single argument (xp)
    metric.add_state_default_factory("a", lambda xp: xp.asarray(0), None)  # type: ignore
    reduce_fn = metric._reductions["a"]
    assert reduce_fn is None, "Saved reduction function is not None."
    assert metric._default_factories.get("a") is not None, (
        "Default factory was not correctly created."
    )

    # default_factory is 'list'
    metric.add_state_default_factory("b", list)  # type: ignore
    assert (
        metric._default_factories.get("b") == list  # noqa: E721
    ), "Default factory should be 'list'."

    # dist_reduce_fn is "sum"
    metric.add_state_default_factory("c", lambda xp: xp.asarray(0), "sum")  # type: ignore
    reduce_fn = metric._reductions["c"]
    assert callable(reduce_fn), "Saved reduction function is not callable."
    assert reduce_fn is dim_zero_sum, (
        "Saved reduction function is not the same as the one used to create the state."
    )
    assert reduce_fn(anp.asarray([1, 1])) == anp.asarray(
        2,
    ), "Saved reduction function does not work as expected."

    # dist_reduce_fn is "mean"
    metric.add_state_default_factory("d", lambda xp: xp.asarray(0), "mean")  # type: ignore
    reduce_fn = metric._reductions["d"]
    assert callable(reduce_fn), "Saved reduction function is not callable."
    assert reduce_fn is dim_zero_mean, (
        "Saved reduction function is not the same as the one used to create the state."
    )
    assert np.allclose(
        reduce_fn(anp.asarray([1.0, 2.0])),
        1.5,
    ), "Saved reduction function does not work as expected."

    # dist_reduce_fn is "cat"
    metric.add_state_default_factory("e", lambda xp: xp.asarray(0), "cat")  # type: ignore
    reduce_fn = metric._reductions["e"]
    assert callable(reduce_fn), "Saved reduction function is not callable."
    assert reduce_fn is dim_zero_cat, (
        "Saved reduction function is not the same as the one used to create the state."
    )
    np.testing.assert_array_equal(
        reduce_fn([anp.asarray([1]), anp.asarray([1])]),
        anp.asarray([1, 1]),
        err_msg="Saved reduction function does not work as expected.",
    )

    # dist_reduce_fn is "max"
    metric.add_state_default_factory("f", lambda xp: xp.asarray(0), "max")  # type: ignore
    reduce_fn = metric._reductions["f"]
    assert callable(reduce_fn), "Saved reduction function is not callable."
    assert reduce_fn is dim_zero_max, (
        "Saved reduction function is not the same as the one used to create the state."
    )
    np.testing.assert_array_equal(
        reduce_fn(anp.asarray([1, 2])),
        anp.asarray(2),
        err_msg="Saved reduction function does not work as expected.",
    )

    # dist_reduce_fn is "min"
    metric.add_state_default_factory("g", lambda xp: xp.asarray(0), "min")  # type: ignore
    metric._add_states(anp)
    reduce_fn = metric._reductions["g"]
    assert callable(reduce_fn), "Saved reduction function is not callable."
    assert reduce_fn is dim_zero_min, (
        "Saved reduction function is not the same as the one used to create the state."
    )
    np.testing.assert_array_equal(
        reduce_fn(anp.asarray([1, 2])),
        anp.asarray(1),
        err_msg="Saved reduction function does not work as expected.",
    )

    # custom reduction function
    def custom_fn(_):
        return anp.asarray(-1)

    metric.add_state_default_factory("h", lambda xp: xp.asarray(0), custom_fn)  # type: ignore
    assert metric._reductions["h"](anp.asarray([1, 1])) == anp.asarray(-1)  # type: ignore

    # test that default values are set correctly
    metric._add_states(anp)
    for name in "abcdefgh":
        default = metric._defaults.get(name, None)
        assert default is not None, f"Default value for {name} is None."
        if apc.is_array_api_obj(default):
            np.testing.assert_array_equal(
                default,
                anp.asarray(0),
                err_msg=f"Default value for {name} is not 0.",
            )
        else:
            assert default == []

        assert hasattr(metric, name), f"Metric does not have attribute {name}."
        attr_val = getattr(metric, name)
        if apc.is_array_api_obj(default):
            np.testing.assert_array_equal(
                attr_val,
                anp.asarray(0),
                err_msg=f"Attribute {name} is not 0.",
            )
        else:
            assert attr_val == []


def test_add_state_default_factory_invalid_input():
    """Test that `add_state_default_factory` method raises errors as expected."""
    metric = DummyMetric()
    with pytest.raises(
        ValueError,
        match="`dist_reduce_fn` must be callable or one of .*",
    ):
        metric.add_state_default_factory("h1", lambda xp: xp.asarray(0), "xyz")  # type: ignore

    with pytest.raises(
        ValueError,
        match="`dist_reduce_fn` must be callable or one of .*",
    ):
        metric.add_state_default_factory("h2", lambda xp: xp.asarray(0), 42)  # type: ignore

    with pytest.raises(
        TypeError,
        match="Expected `default_factory` to be a callable, but got <class 'list'>.",
    ):
        metric.add_state_default_factory("h3", [lambda xp: xp.asarray(0)], "sum")  # type: ignore

    with pytest.raises(
        TypeError,
        match="Expected `default_factory` to be a callable, but got <class 'int'>.",
    ):
        metric.add_state_default_factory("h4", 42, "sum")  # type: ignore

    def custom_fn(xp, _):
        return xp.asarray(-1)

    with pytest.raises(
        TypeError,
        match="Expected `default_factory` to be a function that takes at most .*",
    ):
        metric.add_state_default_factory("h5", custom_fn)  # type: ignore

    with pytest.raises(
        ValueError,
        match="Argument `name` must be a valid Python identifier. Got `h6!`.",
    ):
        metric.add_state_default_factory("h6!", list)  # type: ignore


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_to_device():
    """Test that `to_device` method works as expected."""
    metric = DummyMetric()
    assert metric.device == "cpu"

    metric = metric.to_device("cuda")
    assert metric.device == "cuda"
    metric.update(torch.tensor(42, device="cuda"))  # type: ignore
    assert metric.x.device.type == "cuda"  # type: ignore

    metric = metric.to_device("cpu")
    assert metric.device == "cpu"

    metric = metric.to_device("cuda")
    assert metric.device == "cuda"
    metric.reset()
    assert metric.x.device.type == "cuda"  # type: ignore

    metric = DummyListStateMetric()
    assert metric.device == "cpu"

    metric = metric.to_device("cuda")
    metric.update(torch.tensor(1.0).to("cuda"))  # type: ignore
    metric.compute()
    torch.testing.assert_close(metric.x, [torch.tensor(1.0, device="cuda")])  # type: ignore

    metric = metric.to_device("cpu")
    torch.testing.assert_close(metric.x, [torch.tensor(1.0, device="cpu")])  # type: ignore
    metric.to_device("cuda")
    torch.testing.assert_close(metric.x, [torch.tensor(1.0, device="cuda")])  # type: ignore


def test_update():
    """Test that `update` method works as expected."""
    metric = DummyMetric()
    metric._add_states(anp)
    assert metric.state_vars == {"x": anp.asarray(0, dtype=anp.float32)}
    assert metric._computed is None
    metric.update(anp.asarray(1, dtype=anp.float32))
    assert metric._computed is None
    assert metric.state_vars == {"x": anp.asarray(1, dtype=anp.float32)}
    metric.update(anp.asarray(2, dtype=anp.float32))
    assert metric.state_vars == {"x": anp.asarray(3, dtype=anp.float32)}
    assert metric._computed is None

    metric = DummyListStateMetric()
    metric._add_states(anp)
    assert metric.state_vars == {"x": []}
    assert metric._computed is None
    metric.update(anp.asarray(1))
    assert metric._computed is None
    assert metric.state_vars == {"x": [anp.asarray(1)]}
    metric.update(anp.asarray(2))
    assert metric.state_vars == {"x": [anp.asarray(1), anp.asarray(2)]}
    assert metric._computed is None


def test_compute():
    """Test that `compute` method works as expected."""
    metric = DummyMetric()

    with pytest.raises(
        RuntimeError,
        match="The `compute` method of DummyMetric was called before the `update`.*",
    ):
        metric.compute()

    metric.update(anp.asarray(1, dtype=anp.float32))
    expected_value = anp.asarray(1, dtype=anp.float32)
    assert metric._computed is None
    np.testing.assert_array_equal(metric.compute(), expected_value)
    np.testing.assert_array_equal(metric._computed, expected_value)
    assert metric.state_vars == {"x": expected_value}

    metric.update(anp.asarray(2, dtype=anp.float32))
    expected_value = anp.asarray(3, dtype=anp.float32)
    assert metric._computed is None
    np.testing.assert_array_equal(metric.compute(), expected_value)
    np.testing.assert_array_equal(metric._computed, expected_value)
    assert metric.state_vars == {"x": expected_value}

    # called without update, should return cached value
    metric._computed = anp.asarray(42, dtype=anp.float32)
    np.testing.assert_array_equal(
        metric.compute(),
        anp.asarray(42, dtype=anp.float32),
    )
    assert metric.state_vars == {"x": anp.asarray(3, dtype=anp.float32)}


def test_reset():
    """Test that reset method works as expected."""

    class A(DummyMetric):
        pass

    class B(DummyListStateMetric):
        pass

    metric = A()
    metric._add_states(anp)
    assert metric.x == anp.asarray(0, dtype=anp.float32)  # type: ignore
    metric.x = anp.asarray(42)  # type: ignore
    metric.reset()
    assert metric.x == anp.asarray(0, dtype=anp.float32)  # type: ignore

    metric = B()
    metric._add_states(anp)
    assert isinstance(metric.x, list)  # type: ignore
    assert len(metric.x) == 0  # type: ignore
    metric.x = anp.asarray(42)  # type: ignore
    metric.reset()
    assert isinstance(metric.x, list)  # type: ignore
    assert len(metric.x) == 0  # type: ignore


def test_reset_compute():
    """Test that `reset`+`compute` methods works as expected."""
    metric = DummyMetric()

    metric.update(anp.asarray(42, dtype=anp.float32))
    assert metric.state_vars == {"x": anp.asarray(42, dtype=anp.float32)}
    np.testing.assert_array_equal(
        metric.compute(),
        anp.asarray(42, dtype=anp.float32),
    )
    metric.reset()
    assert metric.state_vars == {}


def test_error_on_compute_before_update():
    """Test that `compute` method raises error when called before `update`."""
    metric = DummyMetric()

    with pytest.raises(
        RuntimeError,
        match="The `compute` method of DummyMetric was called before the `update`.*",
    ):
        metric.compute()

    # after update, should work
    metric.update(anp.asarray(42, dtype=anp.float32))
    result = metric.compute()
    np.testing.assert_array_equal(result, anp.asarray(42, dtype=anp.float32))


def test_clone():
    """Test the `clone` method."""
    metric = DummyMetric()
    metric_clone = metric.clone()
    assert metric is not metric_clone
    assert metric.state_vars is not metric_clone.state_vars
    assert metric._default_factories is not metric_clone._default_factories
    assert metric._reductions is not metric_clone._reductions

    metric.update(anp.asarray(42, dtype=anp.float32))
    assert metric.state_vars == {
        "x": anp.asarray(42, dtype=anp.float32),
    } and not hasattr(metric_clone, "x")
    assert metric._update_count == 1 and metric_clone._update_count == 0

    metric_clone = metric.clone()
    assert metric is not metric_clone
    assert metric.state_vars == metric_clone.state_vars
    assert metric._update_count == metric_clone._update_count
    assert metric._computed == metric_clone._computed

    metric.compute()
    assert (
        anp.all(metric._computed == anp.asarray(42, dtype=anp.float32))
        and metric_clone._computed is None
    )
    metric_clone = metric.clone()
    assert metric is not metric_clone
    assert metric.state_vars == metric_clone.state_vars
    assert anp.all(metric._computed == metric_clone._computed)


def test_call():
    """Test that the `__call__` method works as expected."""
    metric = DummyMetric()
    assert metric.state_vars == {}
    assert metric._computed is None

    metric(anp.asarray(42, dtype=anp.float32))
    assert metric.state_vars == {"x": anp.asarray(42, dtype=anp.float32)}
    assert metric._computed is None

    metric(anp.asarray(1, dtype=anp.float32))
    assert metric.state_vars == {"x": anp.asarray(43, dtype=anp.float32)}
    assert metric._computed is None

    metric.reset()
    assert metric.state_vars == {}
    assert metric._computed is None


@pytest.mark.parametrize("method", ["call", "update"])
@pytest.mark.parametrize("metric", [DummyMetric, DummyListStateMetric])
def test_update_count_torch(metric, method):
    """Test that `_update_count` attribute is correctly updated."""
    m = metric()
    x = torch.randn(
        1,
    ).squeeze()
    for i in range(10):
        if method == "update":
            m.update(x)
        if method == "call":
            _ = m(x)
        assert m._update_count == i + 1

    m.reset()
    assert m._update_count == 0
