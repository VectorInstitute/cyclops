"""Base class for all metrics."""
import functools
import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.distributed_backends import get_backend
from cyclops.evaluate.metrics.experimental.utils.ops import (
    apply_to_array_collection,
    clone,
    dim_zero_cat,
    dim_zero_max,
    dim_zero_mean,
    dim_zero_min,
    dim_zero_sum,
    flatten_seq,
)
from cyclops.evaluate.metrics.experimental.utils.typing import Array
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)

_METRIC_REGISTRY = {}

TState = Union[Array, List[Array]]


@runtime_checkable
class StateFactory(Protocol):
    """Protocol for a function that creates a state variable."""

    def __call__(self, xp: Optional[Any] = None) -> TState:
        """Create a state variable."""
        ...


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the metric."""
        self.update_state: Callable[..., Any] = self._wrap_update(  # type: ignore
            self.update_state,
        )
        self.compute: Callable[..., Any] = self._wrap_compute(  # type: ignore
            self.compute,
        )

        dist_backend = kwargs.get("dist_backend", "non_distributed")
        self.dist_backend = get_backend(dist_backend)

        self._device = "cpu"
        self._update_count: int = 0
        self._computed: Any = None
        self._default_factories: Dict[str, StateFactory] = {}
        self._defaults: Dict[str, TState] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}

        self._is_synced = False
        self._cache: Optional[Dict[str, TState]] = None

    def __init_subclass__(
        cls: Any,
        registry_key: Optional[str] = None,
        force_register: bool = False,
        **kwargs: Any,
    ):
        """Add subclass to the metric registry."""
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

    @property
    def device(self) -> Union[str, Any]:
        """Return the device where the metric states are stored."""
        return self._device

    @property
    def state(self) -> Dict[str, TState]:
        """Return the state of the metric."""
        return {attr: getattr(self, attr) for attr in self._defaults}

    def add_state_factory(
        self,
        name: str,
        default_factory: StateFactory,
        dist_reduce_fn: Optional[Union[str, Callable[..., Any]]] = None,
    ) -> None:
        """Add a factory function for creating a state variable.

        Parameters
        ----------
        name : str
            The name of the state variable.
        default_factory : Callable[..., Union[Array, List[Array]]]
            A function that creates the state variable. The function can take
            no arguments or exactly one argument named `xp` (the array API namespace)
            and must return an array-API-compatible object or a list of
            array-API-compatible objects.
        dist_reduce_fn : str or Callable[..., Any], optional
            The function to use to reduce the state variable across all processes.
            If `None`, no reduction will be performed. If a string, the string
            must be one of ['mean', 'sum', 'cat', 'min', 'max']. If a callable,
            the callable must take a single argument (the state variable) and
            return a reduced version of the state variable.

        """
        if not name.isidentifier():
            raise ValueError(
                f"Argument `name` must be a valid python identifier, but got `{name}`.",
            )
        if not callable(default_factory):
            raise TypeError(
                "Expected `default_factory` to be a callable, but got "
                f"{type(default_factory)}.",
            )

        params = inspect.signature(default_factory).parameters
        check = (
            isinstance(default_factory, StateFactory)
            and default_factory.__name__ == "list"  # type: ignore
            or (
                len(params) == 0
                or (len(params) == 1 and list(params.keys())[0] == "xp")
            )
        )
        if not check:
            raise TypeError(
                "Expected `default_factory` to be a function that takes at most "
                "one argument named 'xp' (the array API namespace), but got "
                f"{inspect.signature(default_factory)}.",
            )

        if dist_reduce_fn == "sum":
            dist_reduce_fn = dim_zero_sum
        elif dist_reduce_fn == "mean":
            dist_reduce_fn = dim_zero_mean
        elif dist_reduce_fn == "max":
            dist_reduce_fn = dim_zero_max
        elif dist_reduce_fn == "min":
            dist_reduce_fn = dim_zero_min
        elif dist_reduce_fn == "cat":
            dist_reduce_fn = dim_zero_cat
        elif dist_reduce_fn is not None and not callable(dist_reduce_fn):
            raise ValueError(
                "`dist_reduce_fn` must be callable or one of "
                "['mean', 'sum', 'cat', 'min', 'max', None]",
            )

        self._default_factories[name] = default_factory
        self._reductions[name] = dist_reduce_fn

    def _add_states(self, xp: Any) -> None:
        """Add the state variables as attributes using the factory functions."""
        for name, factory in self._default_factories.items():
            params = inspect.signature(factory).parameters
            if len(params) == 1 and list(params.keys())[0] == "xp":
                value = factory(xp=xp)
            else:
                value = factory()

            _validate_state_variable_type(name, value)

            setattr(self, name, value)
            self._defaults[name] = (
                clone(value) if apc.is_array_api_obj(value) else deepcopy(value)
            )

    @abstractmethod
    def update_state(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of the metric."""

    @abstractmethod
    def compute(self) -> Any:
        """Compute the final value of the metric from the state variables."""

    def reset_state(self) -> None:
        """Reset the state variables of the metric to the default values."""
        for state_name, default_value in self._defaults.items():
            if apc.is_array_api_obj(default_value):
                setattr(
                    self,
                    state_name,
                    apc.to_device(clone(default_value), self.device),
                )
            elif isinstance(default_value, list):
                setattr(
                    self,
                    state_name,
                    [
                        apc.to_device(clone(array), self.device)
                        for array in default_value
                    ],
                )
            else:
                raise TypeError(
                    "The value of state variable must be an array API-compatible "
                    "object or a list of array API-compatible objects. "
                    f"But got {state_name}={default_value} instead.",
                )

        self._update_count = 0
        self._computed = None
        self._cache = None
        self._is_synced = False

    def clone(self) -> "Metric":
        """Return a copy of the metric."""
        return deepcopy(self)

    def to_device(
        self,
        device: str,
        stream: Optional[Union[int, Any]] = None,
    ) -> "Metric":
        """Move the state variables of the metric to the given device.

        Parameters
        ----------
        device : str
            The device to move the state variables to.
        stream : int or Any, optional
            The stream to use when moving the state variables to the device.
        """
        for name in self._defaults:
            value = getattr(self, name)
            _validate_state_variable_type(name, value)

            if apc.is_array_api_obj(value):
                setattr(self, name, apc.to_device(value, device, stream=stream))
            elif isinstance(value, list):
                setattr(
                    self,
                    name,
                    [apc.to_device(array, device, stream=stream) for array in value],
                )

        self._device = device
        return self

    def sync(self) -> None:
        """Synchronzie the metric states across all processes.

        This method is a no-op if the distributed backend is not initialized or
        if using a non-distributed backend.
        """
        if self._is_synced:
            raise RuntimeError("The Metric has already been synced.")

        if not self.dist_backend.is_initialized:
            if self.dist_backend.world_size == 1:
                self._is_synced = True

                # cache prior to syncing
                self._cache = {attr: getattr(self, attr) for attr in self._defaults}
            return

        # cache prior to syncing
        self._cache = {attr: getattr(self, attr) for attr in self._defaults}

        # sync
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of
            # all_gather operations
            if (
                reduction_fn == dim_zero_cat
                and isinstance(input_dict[attr], list)
                and len(input_dict[attr]) > 1
            ):
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        output_dict = apply_to_array_collection(
            input_dict,
            self.dist_backend.all_gather,
        )

        for attr, reduction_fn in self._reductions.items():
            if isinstance(output_dict[attr], list) and len(output_dict[attr]) == 0:
                setattr(self, attr, [])
                continue

            # stack or flatten inputs before reduction
            first_elem = output_dict[attr][0]
            if apc.is_array_api_obj(first_elem):
                xp = apc.array_namespace(first_elem)
                output_dict[attr] = xp.stack(output_dict[attr])
            elif isinstance(first_elem, list):
                output_dict[attr] = flatten_seq(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("`reduction_fn` must be callable or None")

            reduced = (
                reduction_fn(output_dict[attr])
                if reduction_fn is not None
                else output_dict[attr]
            )
            setattr(self, attr, reduced)

        self._is_synced = True

    def unsync(self) -> None:
        """Un-sync the metric states across all processes.

        Restore cached local state variables of the object.
        """
        if not self._is_synced:
            raise RuntimeError(
                "The Metric has already been un-synced. "
                "This may be because the distributed backend is not initialized.",
            )

        if self._cache is None:
            raise RuntimeError(
                "The internal cache should exist to unsync the Metric. "
                "This is likely because the distributed backend is not initialized.",
            )

        # if we synced, restore to cache so that we can continue to accumulate
        # un-synced state
        for attr, val in self._cache.items():
            setattr(self, attr, val)

        self._is_synced = False
        self._cache = None

    @contextmanager
    def _sync_context(self) -> Generator[None, None, None]:
        """Sync and unsync the metric states within a context manager."""
        self.sync()

        yield

        self.unsync()

    def _wrap_update(self, update: Callable[..., None]) -> Callable[..., None]:
        """Wrap the update function.

        Enusres that the state varibales are created using the factory functions
        before the first call to `update_state`. Also ensures that the state variables
        are moved to the device of the first array API-compatible object passed to
        `update_state`.
        Tracks the number of times `update_state` is called and resets the cached
        result of `compute` if `update_state` is called again.
        """

        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            if (
                not bool(self._defaults) and bool(self._default_factories)
            ) or self._default_factories.keys() != self._defaults.keys():
                arrays = [obj for obj in args if apc.is_array_api_obj(obj)]
                arrays.extend(
                    (obj for obj in kwargs.values() if apc.is_array_api_obj(obj)),
                )
                if len(arrays) == 0:
                    raise ValueError(
                        f"The `update` method of metric {self.__class__.__name__} "
                        "was called without any array API-compatible objects. "
                        "This may lead to errors as metric state variables may "
                        "not yet be defined.",
                    )
                xp = apc.get_namespace(*arrays)
                self._add_states(xp)

                # move state variables to device of first array
                device = apc.device(arrays[0])
                self.to_device(device)

            self._computed = None
            self._update_count += 1

            return update(*args, **kwargs)

        return wrapped_func

    def _wrap_compute(self, compute: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap the compute function.

        Synchoronizes the metric states across all processes before computing the
        final value of the metric. Also returns the cached result if the metric
        has already been computed, avoiding redundant computation.
        """

        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if self._update_count == 0:
                raise RuntimeError(
                    f"The `compute` method of {self.__class__.__name__} was called "
                    "before the `update` method. This will lead to errors, "
                    "as the state variables have not yet been initialized.",
                )

            if self._computed is not None:
                return self._computed  # return cached result

            with self._sync_context():
                value = compute(*args, **kwargs)

            self._computed = value

            return value

        return wrapped_func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Update the global metric state and compute the metric value for a batch."""
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
        self._is_synced = False

        return batch_result

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "Metric":
        """Deepcopy the metric.

        This is needed because the metric may contain array API objects that don
        not allow Array objects to be instantiated directly using the `__new__`
        method. An example of this is the `Array` object in the `numpy.array_api`
        namespace.
        """
        cls = self.__class__
        obj_copy = cls.__new__(cls)

        if memo is None:
            memo = {}
        memo[id(self)] = obj_copy

        for k, v in self.__dict__.items():
            if k == "_cache" and v is not None:
                _cache_ = apply_to_array_collection(
                    v,
                    lambda x: apc.to_device(clone(x), self.device),
                )
                setattr(obj_copy, k, _cache_)
            elif k == "_defaults" and v is not None:
                _defaults_ = apply_to_array_collection(
                    v,
                    lambda x: apc.to_device(clone(x), self.device),
                )
                setattr(obj_copy, k, _defaults_)
            elif apc.is_array_api_obj(v):
                setattr(obj_copy, k, apc.to_device(clone(v), self.device))
            else:
                setattr(obj_copy, k, deepcopy(v, memo))
        return obj_copy

    def __repr__(self) -> str:
        """Return a string representation of the metric."""
        return f"{self.__class__.__name__}"

    def __abs__(self) -> "Metric":
        """Return the absolute value of the metric."""
        return OperatorMetric("__abs__", self, None)

    def __add__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Add two metrics, a metric and a scalar or a metric and an array."""
        return OperatorMetric("__add__", self, other)

    def __and__(self, other: Union[bool, int, "Metric", Array]) -> "Metric":
        """Compute the bitwise AND of a metric and another object."""
        return OperatorMetric("__and__", self, other)

    def __eq__(self, other: Union[bool, float, int, "Metric", Array]) -> Array:
        """Compare the metric to another object for equality."""
        return OperatorMetric("__eq__", self, other)

    def __floordiv__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Compute the floor division of a metric and another object."""
        return OperatorMetric("__floordiv__", self, other)

    def __ge__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Compute the truth value of `self` >= `other`."""
        return OperatorMetric("__ge__", self, other)

    def __gt__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Compute the truth value of `self` > `other`."""
        return OperatorMetric("__gt__", self, other)

    def __invert__(self) -> "Metric":
        """Compute the bitwise NOT of the metric."""
        return OperatorMetric("__invert__", self, None)

    def __le__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Compute the truth value of `self` <= `other`."""
        return OperatorMetric("__le__", self, other)

    def __lt__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Compute the truth value of `self` < `other`."""
        return OperatorMetric("__lt__", self, other)

    def __matmul__(self, other: Union["Metric", Array]) -> "Metric":
        """Matrix multiply two metrics or a metric and an array."""
        return OperatorMetric("__matmul__", self, other)

    def __mod__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Compute the remainder when a metric is divided by another object."""
        return OperatorMetric("__mod__", self, other)

    def __mul__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Multiply two metrics, a metric and a scalar or a metric and an array."""
        return OperatorMetric("__mul__", self, other)

    def __ne__(self, other: Union[bool, float, int, "Metric", Array]) -> Array:
        """Compute the truth value of `self` != `other`."""
        return OperatorMetric("__ne__", self, other)

    def __neg__(self) -> "Metric":
        """Negate every element of the metric result."""
        return OperatorMetric("__neg__", self, None)

    def __or__(self, other: Union[bool, int, "Metric", Array]) -> "Metric":
        """Evaluate `self` | `other`."""
        return OperatorMetric("__or__", self, other)

    def __pos__(self) -> "Metric":
        """Evaluate `+self` for every element of the metric result."""
        return OperatorMetric("__pos__", self, None)

    def __pow__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Raise the metric to the power of another object."""
        return OperatorMetric("__pow__", self, other)

    def __sub__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Subtract two metrics, a metric and a scalar or a metric and an array."""
        return OperatorMetric("__sub__", self, other)

    def __truediv__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Divide two metrics, a metric and a scalar or a metric and an array."""
        return OperatorMetric("__truediv__", self, other)

    def __xor__(self, other: Union[bool, int, "Metric", Array]) -> "Metric":
        """Evaluate `self` ^ `other`."""
        return OperatorMetric("__xor__", self, other)


def _validate_state_variable_type(name: str, value: Any) -> None:
    if not apc.is_array_api_obj(value) and not (
        isinstance(value, list) and all(apc.is_array_api_obj(x) for x in value)
    ):
        raise TypeError(
            "The value of state variable must be an array API-compatible "
            "object or a list of array API-compatible objects. "
            f"Got {name}={value} instead.",
        )


class OperatorMetric(Metric):
    """A metric used to apply an operation to one or two metrics.

    Parameters
    ----------
    operator : str
        The operator to apply.
    metric_a : bool, int, float, Metric, Array
        The first metric to apply the operator to.
    metric_b : bool, int, float, Metric, Array, optional
        The second metric to apply the operator to. For unary operators, this
        should be None.

    """

    def __init__(
        self,
        operator: str,
        metric_a: Union[bool, int, float, Metric, Array],
        metric_b: Optional[Union[bool, int, float, Metric, Array]],
    ) -> None:
        """Initialize the metric."""
        super().__init__()

        self._op = operator
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
        result_a = (
            self.metric_a.compute()
            if isinstance(self.metric_a, Metric)
            else self.metric_a
        )

        result_b = (
            self.metric_b.compute()
            if isinstance(self.metric_b, Metric)
            else self.metric_b
        )

        if self.metric_b is None:
            return getattr(result_a, self._op)()

        return getattr(result_a, self._op)(result_b)

    def reset_state(self) -> None:
        """Reset the state of each metric."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset_state()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset_state()

    def to_device(
        self,
        device: str,
        stream: Optional[Union[int, Any]] = None,
    ) -> Metric:
        """Move the state variables of the metric to the given device."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.to_device(device, stream=stream)
        elif apc.is_array_api_obj(self.metric_a):
            apc.to_device(self.metric_a, device, stream=stream)

        if isinstance(self.metric_b, Metric):
            self.metric_b.to_device(device, stream=stream)
        elif apc.is_array_api_obj(self.metric_b):
            apc.to_device(self.metric_b, device, stream=stream)

        return self

    def _wrap_compute(self, compute: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap the compute function to apply the operator."""
        return compute

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        _op_metrics = f"(\n  {self._op}(\n    {self.metric_a!r},\n    {self.metric_b!r}\n  )\n)"  # noqa: E501
        return self.__class__.__name__ + _op_metrics

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute on the metric or one of its submetrics."""
        # only use after __init__ is done and avoid infinite recursion
        # use __dict__
        if name in ("metric_a", "metric_b", "op"):
            super().__setattr__(name, value)
            return

        attr_set = False
        if (  # bool, int, float don't have __dict__
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
                    f"and {getattr(self.metric_b, name)}.",
                )

        raise AttributeError(
            f"Neither the metric nor its sub-metrics have attribute {name}.",
        )
