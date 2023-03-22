"""Base abstract class for all metrics."""

import functools
import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, UserDict
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from cyclops.evaluate.metrics.utils import (
    _apply_function_recursively,
    _get_value_if_singleton_array,
)
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)

_METRIC_REGISTRY = {}


class Metric(ABC):
    """Abstract base class for metrics classes."""

    def __init__(self) -> None:
        self.update_state: Callable[..., Any] = self._wrap_update(  # type: ignore
            self.update_state
        )  # pylint: disable=method-hidden
        self.compute: Callable[..., Any] = self._wrap_compute(  # type: ignore
            self.compute
        )  # pylint: disable=method-hidden
        self._update_count: int = 0
        self._computed: Any = None
        self._defaults: Dict[str, Union[List[Any], npt.NDArray[Any]]] = {}

    def __init_subclass__(
        cls: Any,
        registry_key: Optional[str] = None,
        force_register: bool = False,
        **kwargs: Any,
    ):
        """Register the subclass in the registry."""
        super().__init_subclass__(**kwargs)

        excluded_classes = ("Metric", "OperatorMetric", "MetricCollection")
        if (  # subclass has not implemented abstract methods
            not (
                cls.update_state is Metric.update_state or cls.compute is Metric.compute
            )
            and cls.__name__ not in excluded_classes
        ) or force_register:
            if registry_key is None:
                LOGGER.warning(
                    "Metric subclass %s has not defined a name. "
                    "It will not be registered in the metric registry.",
                    cls.__name__,
                )
            else:
                if registry_key in _METRIC_REGISTRY:
                    LOGGER.warning(
                        "Metric %s has already been registered. "
                        "It will be overwritten by %s.",
                        registry_key,
                        cls.__name__,
                    )
                _METRIC_REGISTRY[registry_key] = cls

    def add_state(self, name: str, default: Union[List[Any], npt.NDArray[Any]]) -> None:
        """Add a state variable to the metric.

        Parameters
        ----------
        name : str
            The name of the state variable.
        default : Union[List, numpy.ndarray]
            The default value of the state variable.

        Raises
        ------
        AttributeError
            If the state variable already exists.
        TypeError
            If the state variable is not a numpy.ndarray or an empty list.

        """
        if hasattr(self, name):
            raise AttributeError(f"Attribute {name} already exists.")

        if not isinstance(default, (np.ndarray, list)) or (
            isinstance(default, list) and default
        ):
            raise TypeError(
                "State variable must be a numpy array or an empty list (where "
                f"numpy arrays can be appended). Got {type(default)} instead."
            )

        if isinstance(default, np.ndarray):
            default = np.ascontiguousarray(default)
        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)

    def reset_state(self) -> None:
        """Reset the metric to its initial state.

        Sets the ``_update_count`` attribute to 0 and the ``_computed`` attribute to
        None. Also resets the state variables to their default values.

        """
        self._update_count = 0
        self._computed = None

        for attr, default in self._defaults.items():
            if isinstance(default, np.ndarray):
                setattr(self, attr, default.copy())
            else:
                setattr(self, attr, [])

    def clone(self) -> "Metric":
        """Return a copy of the metric."""
        return deepcopy(self)

    @abstractmethod
    def update_state(  # pylint: disable=method-hidden
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Update the state of the metric."""

    @abstractmethod
    def compute(self) -> Any:  # pylint: disable=method-hidden
        """Compute the final value of the metric from the state variables."""

    def _wrap_update(self, update: Callable[..., None]) -> Callable[..., None]:
        """Manage the internal attributes before calling the update method.

        Sets the ``_computed`` attribute to None and increments the ``_update_count``
        attribute before calling the custom update method.

        Parameters
        ----------
        update : Callable
            The update method of the metric.

        Returns
        -------
        wrapped_func : Callable
            The wrapped update method.

        """

        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1

            return update(*args, **kwargs)

        return wrapped_func

    def _wrap_compute(self, compute: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap the ``compute`` method to ensure safety and caching.

        Raises a warning if the ``compute`` method is called before the ``update``
        method has been called at least once. Also caches the result of the
        ``compute`` method to avoid unnecessary recomputation.

        Parameters
        ----------
        compute: Callable
            The compute method of the metric.

        Returns
        -------
        wrapped_func: Callable
            The wrapped compute method.

        Warns
        -----
        If the ``compute`` method has not been called at least once.

        """

        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if self._update_count == 0:
                LOGGER.warning(
                    "The ``compute`` method of metric %s was called before the "
                    "``update`` method which may lead to errors, as metric states "
                    "have not yet been updated.",
                    self.__class__.__name__,
                )

            if self._computed is not None:
                return self._computed  # return cached result

            value = compute(*args, **kwargs)
            self._computed = _apply_function_recursively(
                value, _get_value_if_singleton_array
            )

            return self._computed

        return wrapped_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Update the global metric state and compute the metric for a batch."""
        # global accumulation
        self.update_state(*args, **kwargs)
        update_count = self._update_count
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # batch computation
        self.reset_state()
        self.update_state(*args, **kwargs)
        batch_result = self.compute()

        # restore global state
        for attr, value in cache.items():
            setattr(self, attr, value)
        self._update_count = update_count
        self._computed = None

        return batch_result

    def __repr__(self) -> str:
        """Return a string representation of the metric."""
        return f"{self.__class__.__name__}"

    def __abs__(self) -> "Metric":
        """Absolute value of two metrics together."""
        return OperatorMetric(np.absolute, self, None)

    def __add__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Add two metrics or a metric and a scalar together."""
        return OperatorMetric(np.add, self, other)

    def __and__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Bitwise AND two metrics together."""
        return OperatorMetric(np.bitwise_and, self, other)

    def __eq__(self, other: object) -> bool:
        """Compare two metrics for equality."""
        if not isinstance(other, (bool, int, float, Metric, List, np.ndarray)):
            return NotImplemented
        return OperatorMetric(np.equal, self, other)  # type: ignore[return-value]

    def __floordiv__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Floor division two metrics together."""
        return OperatorMetric(np.floor_divide, self, other)

    def __ge__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Compare two metrics for greater than or equal to."""
        return OperatorMetric(np.greater_equal, self, other)

    def __gt__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Compare two metrics for greater than."""
        return OperatorMetric(np.greater, self, other)

    def __inv__(self) -> "Metric":
        """Invert a metric."""
        return OperatorMetric(np.bitwise_not, self, None)

    def __invert__(self) -> "Metric":
        """Invert a metric."""
        return OperatorMetric(np.bitwise_not, self, None)

    def __le__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Compare two metrics for less than or equal to."""
        return OperatorMetric(np.less_equal, self, other)

    def __lt__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Compare two metrics for less than."""
        return OperatorMetric(np.less, self, other)

    def __matmul__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Matrix multiplication of two metrics together."""
        return OperatorMetric(np.matmul, self, other)

    def __mod__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Modulo two metrics together."""
        return OperatorMetric(np.mod, self, other)

    def __mul__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Multiply two metrics or a metric and a scalar together."""
        return OperatorMetric(np.multiply, self, other)

    def __ne__(self, other: object) -> bool:
        """Compare two metrics for inequality."""
        if not isinstance(other, (bool, int, float, Metric, List, np.ndarray)):
            return NotImplemented
        return OperatorMetric(np.not_equal, self, other)  # type: ignore[return-value]

    def __neg__(self) -> "Metric":
        """Negate a metric."""
        return OperatorMetric(np.negative, self, None)

    def __or__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Bitwise OR two metrics together."""
        return OperatorMetric(np.bitwise_or, self, other)

    def __pos__(self) -> "Metric":
        """Positive value of a metric."""
        return OperatorMetric(np.absolute, self, None)

    def __pow__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Raise two metrics to a power."""
        return OperatorMetric(np.power, self, other)

    def __radd__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Add two metrics or a metric and a scalar together."""
        return OperatorMetric(np.add, other, self)

    def __rand__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Bitwise AND two metrics together."""
        return OperatorMetric(np.bitwise_and, other, self)

    def __rfloordiv__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Floor division two metrics together."""
        return OperatorMetric(np.floor_divide, other, self)

    def __rmatmul__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Matrix multiplication of two metrics together."""
        return OperatorMetric(np.matmul, other, self)

    def __rmod__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Modulo two metrics together."""
        return OperatorMetric(np.mod, other, self)

    def __rmul__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Multiply two metrics or a metric and a scalar together."""
        return OperatorMetric(np.multiply, other, self)

    def __ror__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Bitwise OR two metrics together."""
        return OperatorMetric(np.bitwise_or, other, self)

    def __rpow__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Raise two metrics to a power."""
        return OperatorMetric(np.power, other, self)

    def __rsub__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Subtract two metrics or a metric and a scalar from each other."""
        return OperatorMetric(np.subtract, other, self)

    def __rtruediv__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Divide two metrics together."""
        return OperatorMetric(np.true_divide, other, self)

    def __rxor__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Bitwise XOR two metrics together."""
        return OperatorMetric(np.bitwise_xor, other, self)

    def __sub__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Subtract two or a metric and a scalar from each other."""
        return OperatorMetric(np.subtract, self, other)

    def __truediv__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Divide two metrics or a metric and a scalar together."""
        return OperatorMetric(np.true_divide, self, other)

    def __xor__(
        self, other: Union[bool, int, float, "Metric", npt.ArrayLike]
    ) -> "Metric":
        """Apply a bitwise XOR to the metric and other compatible object."""
        return OperatorMetric(np.bitwise_xor, self, other)


class OperatorMetric(Metric):
    """A metric used to apply an operation to one or two metrics.

    Parameters
    ----------
    operator : Callable[..., Any]
        The operator to apply. A numpy function is recommended.
    metric1 : bool, int, float, Metric, ArrayLike
        The first metric to apply the operator to.
    metric2 : bool, int, float, Metric, ArrayLike, None
        The second metric to apply the operator to. For unary operators, this
        should be None.

    """

    def __init__(
        self,
        operator: Callable[..., Any],
        metric_a: Union[bool, int, float, Metric, npt.ArrayLike],
        metric_b: Union[bool, int, float, Metric, npt.ArrayLike, None],
    ):
        super().__init__()

        self.op = operator  # pylint: disable=invalid-name
        self.metric_a = metric_a
        self.metric_b = metric_b

    def update_state(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of each metric."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.update_state(*args, **kwargs)

        if isinstance(self.metric_b, Metric):
            self.metric_b.update_state(*args, **kwargs)

    def compute(self) -> Any:
        """Compute the value of each metric, then apply the operator."""
        if isinstance(self.metric_a, Metric):
            metric1_result = self.metric_a.compute()
        else:
            metric1_result = self.metric_a

        if isinstance(self.metric_b, Metric):
            metric2_result = self.metric_b.compute()
        else:
            metric2_result = self.metric_b

        if self.metric_b is None:
            return self.op(metric1_result)

        return self.op(metric1_result, metric2_result)

    def reset_state(self) -> None:
        """Reset the state of each metric."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset_state()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset_state()

    def _wrap_compute(self, compute: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap the compute function to apply the operator."""
        return compute

    def __repr__(self) -> str:
        """Return a string representation of the metric."""
        _op_metrics = (
            f"(\n  {self.op.__name__}(\n    {repr(self.metric_a)},\n    "
            f"{repr(self.metric_b)}\n  )\n)"
        )

        return self.__class__.__name__ + _op_metrics

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the metric or one of its submetrics."""
        # only use after __init__ is done and avoid infinite recursion
        # use __dict__
        if name in ("metric_a", "metric_b", "op"):
            super().__setattr__(name, value)
            return

        attr_set = False
        if (  # bool, int, float and arraylikes don't have __dict__
            self.__dict__.get("metric_a") is not None
            and isinstance(self.__dict__["metric_a"], Metric)
            and name in self.__dict__["metric_a"].__dict__
        ):
            setattr(self.metric_a, name, value)
            attr_set = True
        if (
            self.__dict__.get("metric_b") is not None
            and isinstance(self.__dict__["metric_b"], Metric)
            and name in self.__dict__["metric_b"].__dict__
        ):
            setattr(self.metric_b, name, value)
            attr_set = True

        if attr_set:
            return

        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the metric or its sub-metrics."""
        if name not in ("metric_a", "metric_a", "op"):
            # if only one of them is a Metric, return the attribute from that one
            if hasattr(self.metric_a, name) and not isinstance(self.metric_b, Metric):
                return getattr(self.metric_a, name)
            if hasattr(self.metric_b, name) and not isinstance(self.metric_a, Metric):
                return getattr(self.metric_b, name)
            # if they are both Metrics, only return if they both have the attribute
            # with the same value
            if (
                hasattr(self.metric_a, name)
                and hasattr(self.metric_b, name)
                and getattr(self.metric_a, name) == getattr(self.metric_b, name)
            ):
                return getattr(self.metric_a, name)
            # otherwise raise an error telling the user that they are both Metrics
            # and have different values for the attribute
            if hasattr(self.metric_a, name) and hasattr(self.metric_b, name):
                raise AttributeError(
                    f"Both {self.metric_a.__class__.__name__} and "
                    f"{self.metric_b.__class__.__name__} have attribute {name} "
                    f"but they have different values: {getattr(self.metric_a, name)} "
                    f"and {getattr(self.metric_b, name)}."
                )

        raise AttributeError(
            f"Neither the metric nor its sub-metrics have attribute {name}."
        )


class MetricCollection(UserDict[str, Union[Metric, "MetricCollection"]]):
    """A collection of metrics.

    This class is used to group metrics together. It is useful for when you want
    to compute multiple metrics at the same time. It behaves like a dictionary
    where the keys are the names of the metrics and the values are the metrics
    themselves. Internally, it groups metrics with similar states together to
    reduce the number of times the state is updated.

    Parameters
    ----------
    metrics : Union[Metric, Sequence[Metric], Dict[str, Metric]]
        The metrics to add to the collection. This can be a single metric, a
        sequence of metrics, or a dictionary mapping names to metrics.
    *other_metrics : Metric
        Additional metrics to add to the collection. This is only used if
        `metrics` is a single metric or a sequence of metrics.
    prefix : str, optional, default=None
        A prefix to add to the names of the metrics.
    postfix : str, optional, default=None
        A postfix to add to the names of the metrics.

    Raises
    ------
    TypeError
        If `metrics` is not a single metric, a sequence of metrics, or a
        dictionary mapping names to metrics.
    TypeError
        If `prefix` or `postfix` is not a string.
    TypeError
        If `metrics` is a dictionary and `other_metrics` is not empty.

    """

    _metric_groups: Dict[int, List[str]]

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *other_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._check_prefix_postfix(prefix, postfix)
        self.prefix = prefix
        self.postfix = postfix

        self.add_metrics(metrics, *other_metrics)

    def _check_prefix_postfix(
        self, prefix: Optional[str], postfix: Optional[str]
    ) -> None:
        """Check that the prefix and postfix are strings."""
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError(
                f"Expected `prefix` to be a string, but got {type(prefix).__name__}."
            )
        if postfix is not None and not isinstance(postfix, str):
            raise TypeError(
                f"Expected `postfix` to be a string, but got {type(postfix).__name__}."
            )

    def add_metrics(  # pylint: disable=too-many-branches
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *other_metrics: Metric,
    ) -> None:
        """Add metrics to the collection."""
        if isinstance(metrics, Metric):
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            metrics = list(metrics)
            extras: List[Any] = []
            for metric in other_metrics:
                (metrics if isinstance(metric, Metric) else extras).append(metric)

            if extras:
                LOGGER.warning(
                    "Received additional metrics that are not of type `Metric`: %s."
                    "These metrics will be ignored.",
                    extras,
                )
        elif other_metrics:
            raise TypeError(
                "`other_metrics` can only be used with a single `Metric` object "
                "or a sequence of `Metric` objects."
            )

        if isinstance(metrics, dict):
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, (MetricCollection, Metric)):
                    raise TypeError(
                        f"Expected value {metric} for key {name} to be of type "
                        f"`Metric` or `MetricCollection`, but got {type(metric)}."
                    )
                if isinstance(metric, MetricCollection):
                    for sub_name, sub_metric in metric.items():
                        self[f"{name}_{sub_name}"] = sub_metric
                else:
                    self[name] = metric
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (MetricCollection, Metric)):
                    raise TypeError(
                        f"Expected metric {metric} to be of type `Metric` or "
                        f"`MetricCollection`, but got {type(metric)}."
                    )
                if isinstance(metric, MetricCollection):
                    for name, sub_metric in metric.items():
                        self[name] = sub_metric
                else:
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(
                            f"Metric {metric} has the same name as another metric "
                            f"in the collection: {name}."
                        )
                    self[name] = metric
        else:
            raise TypeError(
                f"Expected `metrics` to be of type `Metric`, `Sequence[Metric]` or "
                f"`Dict[str, Metric]`, but got {type(metrics)}."
            )

        self._group_metrics()

    def update_state(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of all metrics in the collection.

        Uses the metric groups to only update the state variables once per group.

        Parameters
        ----------
        *args: Any
            The positional arguments to pass to the update_state method of each metric.
        **kwargs: Any
            The keyword arguments to pass to the update_state method of each metric.

        """
        # find metrics with the same state and update them only once
        for metrics in self._metric_groups.values():
            base_metric = self.data[metrics[0]]
            base_metric.update_state(*args, **kwargs)

        self._create_metric_group_state_ref()

    def compute(self) -> Dict[str, Any]:
        """Compute the metrics in the collection."""
        result = {name: metric.compute() for name, metric in self.items(keep_base=True)}
        result = _flatten_dict(result)
        return {self._set_name(k): v for k, v in result.items()}

    def reset_state(self) -> None:
        """Reset the state of all metrics in the collection."""
        for _, metric in self.items(keep_base=True):
            metric.reset_state()

        self._create_metric_group_state_ref()  # reset the references

    def _create_metric_group_state_ref(self) -> None:
        """Create references to the state variables for metrics in the same group."""
        for metric_names in self._metric_groups.values():
            base_metric = self.data[metric_names[0]]

            for metric_name in metric_names[1:]:
                metric = self.data[metric_name]
                for state in getattr(metric, "_defaults"):
                    setattr(metric, state, getattr(base_metric, state))

                setattr(metric, "_update_count", getattr(base_metric, "_update_count"))

    def _group_metrics(self) -> None:
        """Group metrics by the state variables they use.

        Returns
        -------
        metric_groups: Dict[int, List[str]]
            A dictionary with the group id as the key and a list of metric names
            as the value.

        """
        metrics_by_state: Dict[int, List[str]] = {}
        for name, metric in self.data.items():
            # serialize the state variables and use the hash as the key
            # use JSON, but make sure numpy arrays are converted to lists
            state = hash(
                json.dumps(
                    getattr(metric, "_defaults"),
                    default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                    sort_keys=True,
                )
            )
            if state not in metrics_by_state:
                metrics_by_state[state] = []
            metrics_by_state[state].append(name)

        self._metric_groups = dict(enumerate(metrics_by_state.values()))

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        return name if self.postfix is None else name + self.postfix

    def _to_renamed_ordered_dict(
        self,
    ) -> OrderedDict[str, Union[Metric, "MetricCollection"]]:
        """Return an ordered dict with the renamed keys."""
        ordered_data = OrderedDict()
        for key, value in self.data.items():
            ordered_data[self._set_name(key)] = value
        return ordered_data

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:  # type: ignore
        """Return an iterable of the ModuleDict key.

        Parameters
        ----------
        keep_base : bool
            Whether to add prefix/postfix on the items collection.

        """
        if keep_base:
            return self.data.keys()
        return self._to_renamed_ordered_dict().keys()

    def values(  # type: ignore[override]
        self,
    ) -> Iterable[Union[Metric, "MetricCollection"]]:
        """Return an iterable of the ModuleDict values.

        Parameters
        ----------
        keep_base : bool
            Whether to add prefix/postfix on the collection.

        """
        self._create_metric_group_state_ref()  # update the references
        return self.data.values()

    def items(  # type: ignore[override]
        self, keep_base: bool = False
    ) -> Iterable[Tuple[str, Union[Metric, "MetricCollection"]]]:
        """Return an iterable of the underlying dictionary's items.

        Parameters
        ----------
        keep_base : bool
            Whether to add prefix/postfix on the collection.

        """
        self._create_metric_group_state_ref()  # update the references
        if keep_base:
            return self.data.items()
        return self._to_renamed_ordered_dict().items()

    def clone(
        self, prefix: Optional[str] = None, postfix: Optional[str] = None
    ) -> "MetricCollection":
        """Create a copy of the metric collection.

        Parameters
        ----------
        prefix : str, optional
            Prefix to add to the name of the metric.
        postfix : str, optional
            Postfix to add to the name of the metric.

        Returns
        -------
        MetricCollection
            A copy of the metric collection.

        """
        new_mc = deepcopy(self)
        self._check_prefix_postfix(prefix, postfix)
        if prefix:
            new_mc.prefix = prefix
        if postfix:
            new_mc.postfix = postfix
        return new_mc

    def __getitem__(self, key: str) -> Union[Metric, "MetricCollection"]:
        """Return the metric with the given name."""
        self._create_metric_group_state_ref()  # update the references
        return self.data[key]

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Apply the __call__ method of all metrics in the collection."""
        batch_result = {
            name: metric(*args, **kwargs) for name, metric in self.items(keep_base=True)
        }
        batch_result = _flatten_dict(batch_result)
        return {self._set_name(k): v for k, v in batch_result.items()}


def _flatten_dict(a_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten dict of dicts into single dict."""
    new_dict = {}
    for key, value in a_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_dict[sub_key] = sub_value
        else:
            new_dict[key] = value
    return new_dict
