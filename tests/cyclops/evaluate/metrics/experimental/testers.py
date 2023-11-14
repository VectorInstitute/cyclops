"""Testers for metrics."""
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Type

import array_api_compat as apc
import numpy as np

from cyclops.evaluate.metrics.experimental.metric import Metric
from cyclops.evaluate.metrics.experimental.utils.ops import clone, flatten
from cyclops.evaluate.metrics.experimental.utils.typing import Array


def _assert_allclose(
    cyclops_result: Any,
    ref_result: Any,
    atol: float = 1e-8,
    key: Optional[str] = None,
) -> None:
    """Recursively assert that two results are within a certain tolerance."""
    if apc.is_array_api_obj(cyclops_result) and apc.is_array_api_obj(ref_result):
        # move to cpu and convert to numpy
        cyclops_result = np.from_dlpack(apc.to_device(cyclops_result, "cpu"))
        ref_result = np.from_dlpack(apc.to_device(ref_result, "cpu"))

        np.testing.assert_allclose(
            cyclops_result,
            ref_result,
            atol=atol,
            equal_nan=True,
        )

    # multi output comparison
    elif isinstance(cyclops_result, Sequence):
        for cyc_res, ref_res in zip(cyclops_result, ref_result):
            _assert_allclose(cyc_res, ref_res, atol=atol)
    elif isinstance(cyclops_result, dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        _assert_allclose(cyclops_result[key], ref_result, atol=atol)
    else:
        raise ValueError("Unknown format for comparison")


def _assert_array(cyclops_result: Any, key: Optional[str] = None) -> None:
    """Recursively check that some input only consists of Arrays."""
    if isinstance(cyclops_result, Sequence):
        for res in cyclops_result:
            _assert_array(res)
    elif isinstance(cyclops_result, Dict):
        if key is None:
            raise KeyError("Provide Key for Dict based metric results.")
        assert apc.is_array_api_obj(cyclops_result[key])
    else:
        assert apc.is_array_api_obj(cyclops_result)


def _class_impl_test(  # noqa: PLR0912
    target: Array,
    preds: Array,
    metric_class: Type[Metric],
    reference_metric: Callable[..., Any],
    metric_args: Optional[Dict[str, Any]] = None,
    atol: float = 1e-8,
    device: str = "cpu",
    use_device_for_ref: bool = False,
):
    """Test output of metric class against a reference metric."""
    assert apc.is_array_api_obj(target) and apc.is_array_api_obj(preds), (
        f"`target` and `preds` must be Array API compatible objects, "
        f"got {type(target)} and {type(preds)}."
    )

    t_size = target.shape[0]
    p_size = preds.shape[0]
    assert (
        p_size == t_size
    ), f"`preds` and `target` have different number of samples: {p_size} and {t_size}."
    num_batches = p_size

    # instantiate metric
    metric_args = metric_args or {}
    metric = metric_class(**metric_args)

    # check that the metric can be cloned
    metric_clone = metric.clone()
    assert metric_clone is not metric, "Metric clone should not be the same object."
    assert type(metric_clone) is type(metric), "Metric clone should be the same type."

    # move to device
    metric = metric.to_device(device)
    preds = apc.to_device(preds, device)
    target = apc.to_device(target, device)

    for i in range(num_batches):  # type: ignore
        # compute batch result and aggregate for global result
        cyc_batch_result = metric(target[i, ...], preds[i, ...])

        ref_batch_result = reference_metric(
            target=apc.to_device(
                target[i, ...],
                device if use_device_for_ref else "cpu",
            ),
            preds=apc.to_device(preds[i, ...], device if use_device_for_ref else "cpu"),
        )
        if isinstance(cyc_batch_result, dict):
            for key in cyc_batch_result:
                _assert_allclose(
                    cyc_batch_result,
                    ref_batch_result[key],
                    atol=atol,
                    key=key,
                )
        else:
            _assert_allclose(cyc_batch_result, ref_batch_result, atol=atol)

    # check on all batches on all ranks
    cyc_result = metric.compute()
    if isinstance(cyc_result, dict):
        for key in cyc_result:
            _assert_array(cyc_result, key=key)
    else:
        _assert_array(cyc_result)

    xp = apc.array_namespace(target, preds)
    if preds.ndim == 1 or (preds.ndim == 2 and target.ndim == 1):
        # 0-D binary and multiclass cases
        total_preds = preds
    else:
        total_preds = xp.concat([preds[i, ...] for i in range(num_batches)])  # type: ignore

    if target.ndim > 1:
        total_target = xp.concat([target[i, ...] for i in range(num_batches)])  # type: ignore
    else:
        total_target = target
    ref_result = reference_metric(
        target=apc.to_device(total_target, device if use_device_for_ref else "cpu"),
        preds=apc.to_device(total_preds, device if use_device_for_ref else "cpu"),
    )

    # assert after aggregation
    if isinstance(ref_result, dict):
        for key in ref_result:
            _assert_allclose(cyc_result, ref_result[key], atol=atol, key=key)
    else:
        _assert_allclose(cyc_result, ref_result, atol=atol)


def _function_impl_test(
    target: Array,
    preds: Array,
    metric_function: Callable[..., Any],
    reference_metric: Callable[..., Any],
    metric_args: Optional[Dict[str, Any]] = None,
    atol: float = 1e-8,
    device: str = "cpu",
    use_device_for_ref: bool = False,
):
    """Test output of a metric function against a reference metric."""
    assert apc.is_array_api_obj(target) and apc.is_array_api_obj(preds), (
        f"`target` and `preds` must be Array API compatible objects, "
        f"got {type(target)} and {type(preds)}."
    )

    t_size = target.shape[0]
    p_size = preds.shape[0]
    assert (
        p_size == t_size
    ), f"`preds` and `target` have different number of samples: {p_size} and {t_size}."

    metric_args = metric_args or {}
    metric = partial(metric_function, **metric_args)

    preds = apc.to_device(preds, device)
    target = apc.to_device(target, device)

    num_batches = p_size
    for i in range(num_batches):
        cyclops_result = metric(target[i, ...], preds[i, ...])

        # always compare to reference metric on CPU
        ref_result = reference_metric(
            target=apc.to_device(
                target[i, ...],
                device if use_device_for_ref else "cpu",
            ),
            preds=apc.to_device(preds[i, ...], device if use_device_for_ref else "cpu"),
        )

        _assert_allclose(cyclops_result, ref_result, atol=atol)


class MetricTester:
    """Test class for all metrics."""

    atol: float = 1e-8

    def run_metric_function_implementation_test(
        self,
        target: Array,
        preds: Array,
        metric_function: Callable[..., Any],
        reference_metric: Callable[..., Any],
        metric_args: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        use_device_for_ref: bool = False,
    ):
        """Test output of a metric function against a reference metric.

        Parameters
        ----------
        target : Array
            The target array. Any Array API compatible object is accepted.
        preds : Array
            The predictions array. Any Array API compatible object is accepted.
        metric_function : Callable[..., Any]
            The metric function to test.
        reference_metric : Callable[..., Any]
            The reference metric function.
        metric_args : Dict[str, Any], optional
            The arguments to pass to the metric function.
        device : str, optional, default="cpu"
            The device to compute the metric on.
        use_device_for_ref : bool, optional, default=False
            Whether to compute the reference metric on the same device as `device`.

        """
        return _function_impl_test(
            target=target,
            preds=preds,
            metric_function=metric_function,
            reference_metric=reference_metric,
            metric_args=metric_args,
            atol=self.atol,
            device=device,
            use_device_for_ref=use_device_for_ref,
        )

    def run_metric_class_implementation_test(
        self,
        target: Array,
        preds: Array,
        metric_class: Type[Metric],
        reference_metric: Callable[..., Any],
        metric_args: Optional[dict] = None,
        device: str = "cpu",
        use_device_for_ref: bool = False,
    ):
        """Test output of a metric class against a reference metric.

        Parameters
        ----------
        target : Array
            The target array. Any Array API compatible object is accepted.
        preds : Array
            The predictions array. Any Array API compatible object is accepted.
        metric_class : Metric
            The metric class to test.
        reference_metric : Callable[..., Any]
            The reference metric function.
        metric_args : Optional[dict], optional
            The arguments to pass to the metric function.
        device : str, optional, default="cpu"
            The device to compute the metric on.
        use_device_for_ref : bool, optional, default=False
            Whether to compute the reference metric on the same device as `device`.
        """
        return _class_impl_test(
            target=target,
            preds=preds,
            metric_class=metric_class,
            reference_metric=reference_metric,
            metric_args=metric_args,
            atol=self.atol,
            device=device,
            use_device_for_ref=use_device_for_ref,
        )


class DummyMetric(Metric):
    """Dummy metric for testing core components."""

    name = "Dummy"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state_default_factory(
            "x",
            lambda xp: xp.asarray(0.0, dtype=xp.float32, device=self.device),  # type: ignore
            dist_reduce_fn="sum",
        )

    def _update_state(self, x: Array) -> None:
        """Update state."""
        self.x += x  # type: ignore

    def _compute_metric(self) -> Array:
        """Compute value."""
        return self.x  # type: ignore


class DummyListStateMetric(Metric):
    """Dummy metric with list state for testing core components."""

    name = "DummyListState"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state_default_factory("x", list, dist_reduce_fn="cat")  # type: ignore

    def _update_state(self, x: Array):
        """Update state."""
        self.x.append(apc.to_device(x, self.device))  # type: ignore

    def _compute_metric(self):
        """Compute value."""
        return self.x  # type: ignore


def _inject_ignore_index(array, ignore_index):
    """Inject ignore index into array."""
    if ignore_index is None:
        return array

    if isinstance(ignore_index, int):
        ignore_index = (ignore_index,)

    if any(any(flatten(array) == idx) for idx in ignore_index):
        return array

    xp = apc.array_namespace(array)
    classes = xp.unique_values(array)

    # select random indices (same size as ignore_index) and set them to ignore_index
    indices = np.random.randint(0, apc.size(array), size=len(ignore_index))  # type: ignore
    array = clone(array)

    # use loop + basic indexing to set ignore_index
    for idx, ignore_idx in zip(indices, ignore_index):  # type: ignore
        xp.reshape(array, (-1,))[idx] = ignore_idx

    # if all classes are removed, add one back
    batch_size = array.shape[0] if array.ndim > 1 else 1
    for i in range(batch_size):
        batch = array[i, ...] if array.ndim > 1 else array
        new_classes = xp.unique_values(batch)
        class_not_in = [c not in new_classes for c in classes]

        if any(class_not_in):
            missing_class = int(np.where(class_not_in)[0][0])
            mask = xp.zeros_like(batch, dtype=xp.bool, device=apc.device(batch))
            for idx in ignore_index:
                mask = xp.logical_or(mask, batch == idx)
            ignored_idx = np.where(mask)[0]
            if len(ignored_idx) > 0:
                batch[int(ignored_idx[0])] = missing_class

    return array
