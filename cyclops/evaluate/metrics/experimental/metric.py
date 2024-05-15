"""Base class for all metrics."""

import inspect
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
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
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)

_METRIC_REGISTRY = {}

State = Union[Array, List[Array]]


@runtime_checkable
class StateFactory(Protocol):
    """Protocol for a function that creates a metric state."""

    def __call__(self, xp: Optional[ModuleType] = None) -> State:
        """Create a metric state."""
        ...


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, **kwargs: Any) -> None:
        dist_backend = kwargs.get("dist_backend", "non_distributed")
        self.dist_backend = get_backend(dist_backend)

        self._device = "cpu"
        self._update_count: int = 0
        self._computed: Any = None
        self._default_factories: Dict[str, StateFactory] = {}
        self._defaults: Dict[str, State] = {}
        self._reductions: Dict[str, Union[str, Callable[..., Any], None]] = {}

        self._is_synced = False
        self._cache: Optional[Dict[str, State]] = None

    def __init_subclass__(
        cls: Any,
        registry_key: Optional[str] = None,
        force_register: bool = False,
    ) -> None:
        """Add subclass to the metric registry."""
        super().__init_subclass__()

        if registry_key is None and force_register:
            warnings.warn(
                "A registry key must be provided when `force_register` is True. "
                "The registration will be skipped.",
                category=UserWarning,
                stacklevel=2,
            )
            return

        if registry_key is not None and not isinstance(registry_key, str):
            raise TypeError(
                "Expected `registry_key` to be `None` or a string, but got "
                f"{type(registry_key)}.",
            )

        is_abstract_cls = inspect.isabstract(cls)
        excluded_classes = ("OperatorMetric", "MetricCollection")
        if force_register or (
            not (is_abstract_cls or cls.__name__ in excluded_classes)
            and registry_key is not None
        ):
            _METRIC_REGISTRY[registry_key] = cls

    @property
    def device(self) -> Union[str, Any]:
        """Return the device on which the metric states are stored."""
        return self._device

    @property
    def state_vars(self) -> Dict[str, State]:
        """Return the state variables of the metric as a dictionary."""
        return {attr: getattr(self, attr) for attr in self._defaults}

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric.

        This name can be different from the class name and does not need to be
        unique.

        """

    @abstractmethod
    def _update_state(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of the metric."""

    @abstractmethod
    def _compute_metric(self) -> Any:
        """Compute the final value of the metric from the state variables."""

    def _add_states(self, xp: Any) -> None:
        """Add the state variables as attributes using the default factory functions."""
        # raise error if no default factories have been added
        if not self._default_factories:
            warnings.warn(
                f"The metric `{self.__class__.__name__}` has no state variables, "
                "which may lead to unexpected behavior. This is likely because the "
                "`update` method was called before the `add_state_default_factory` "
                "method was called.",
                category=UserWarning,
                stacklevel=2,
            )

        for name, factory in self._default_factories.items():
            params = inspect.signature(factory).parameters
            if len(params) == 1 and list(params.keys())[0] == "xp":
                value = factory(xp=xp)
            else:
                value = factory()

            _validate_state_variable_type(name, value)

            setattr(self, name, value)
            self._defaults[name] = (
                clone(value) if apc.is_array_api_obj(value) else deepcopy(value)  # type: ignore[arg-type]
            )

    def add_state_default_factory(
        self,
        name: str,
        default_factory: StateFactory,
        dist_reduce_fn: Optional[Union[str, Callable[..., Any]]] = None,
    ) -> None:
        """Add a function for creating default values for state variables.

        Parameters
        ----------
        name : str
            The name of the state.
        default_factory : Callable[..., Union[Array, List[Array]]]
            A function that creates the state. The function can take
            no arguments or exactly one argument named `xp` (the array API namespace)
            and must return an array-API-compatible object or a list of
            array-API-compatible objects.
        dist_reduce_fn : str or Callable[..., Any], optional
            The function to use to reduce the state across all processes.
            If `None`, no reduction will be performed. If a string, the string
            must be one of ['mean', 'sum', 'cat', 'min', 'max']. If a callable,
            the callable must take a single argument (the state) and
            return a reduced version of the state.

        """
        if not name.isidentifier():
            raise ValueError(
                f"Argument `name` must be a valid Python identifier. Got `{name}`.",
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

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of the metric.

        This method calls the `_update_state` method, which should be implemented
        by the subclass. The `_update_state` method should update the state variables
        of the metric using the array API-compatible objects passed to this method.
        This method enusres that the state variables are created using the factory
        functions added via the `add_state_default_factory` before the first call to
        `_update_state`. It also ensures that the state variables are moved to the
        device of the first array-API-compatible object passed to it. The method
        tracks the number of times `update` is called and resets the cached result
        of `compute` whenever `update` is called.

        Notes
        -----
        - This method should be called before the `compute` method is called
        for the first time to ensure that the state variables are initialized.
        """
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

            # move state variables to device of first array
            device = apc.device(arrays[0])
            self.to_device(device)

            self._add_states(xp)

        self._computed = None
        self._update_count += 1

        self._update_state(*args, **kwargs)

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
                self._cache = {attr: getattr(self, attr) for attr in self._defaults}
            return

        self._cache = {attr: getattr(self, attr) for attr in self._defaults}

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
        """Restore cached local metric state."""
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

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the final value of the metric from the state variables.

        Prior to calling the `_compute_metric` method, which should be implemented
        by the subclass, this method ensures that the metric states are synced
        across all processes and guards against potentially calling the `compute`
        method before the state variables have been initialized. This method
        also caches the result of the metric computation so that it can be returned
        without recomputing the metric.
        """
        if self._update_count == 0:
            raise RuntimeError(
                f"The `compute` method of {self.__class__.__name__} was called "
                "before the `update` method. This will lead to errors, "
                "as the state variables have not yet been initialized.",
            )

        if self._computed is not None:
            return self._computed  # return cached result

        self.sync()
        value = self._compute_metric(*args, **kwargs)
        self.unsync()

        self._computed = value

        return value

    def reset(self) -> None:
        """Reset the metric state to default values."""
        for state_name, default_value in self._defaults.items():
            if apc.is_array_api_obj(default_value):
                setattr(
                    self,
                    state_name,
                    apc.to_device(clone(default_value), self.device),  # type: ignore[arg-type]
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
                    f"Expected the value of state `{state_name}` to be an array API "
                    "object or a list of array API objects. But got "
                    f"`{type(default_value)} instead.",
                )
        self._defaults = {}

        self._update_count = 0
        self._computed = None
        self._cache = None
        self._is_synced = False

    def clone(self) -> "Metric":
        """Return a deep copy of the metric."""
        return deepcopy(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Update the global metric state and compute the metric value for a batch."""
        # global accumulation
        self.update(*args, **kwargs)
        update_count = self._update_count
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # batch computation
        self.reset()
        self.update(*args, **kwargs)
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
            elif isinstance(v, (list, tuple)):
                seq_var = apply_to_array_collection(
                    v,
                    lambda x: apc.to_device(clone(x), self.device),
                )
                setattr(
                    obj_copy,
                    k,
                    [
                        deepcopy(arr, memo) if not apc.is_array_api_obj(arr) else arr
                        for arr in seq_var
                    ],
                )
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

    def __eq__(  # type: ignore[override]
        self,
        other: Union[bool, float, int, "Metric", Array],  # type: ignore[override]
    ) -> Array:
        """Compare the metric to another object for equality."""
        return OperatorMetric("__eq__", self, other)  # type: ignore[return-value]

    def __floordiv__(
        self,
        other: Union[int, float, "Metric", Array],
    ) -> "Metric":
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

    def __ne__(  # type: ignore[override]
        self,
        other: Union[bool, float, int, "Metric", Array],  # type: ignore[override]
    ) -> Array:
        """Compute the truth value of `self` != `other`."""
        return OperatorMetric("__ne__", self, other)  # type: ignore[return-value]

    def __neg__(self) -> "Metric":
        """Negate every element of the metric result."""
        return OperatorMetric("__neg__", self, None)

    def __or__(self, other: Union[bool, int, "Metric", Array]) -> "Metric":
        """Evaluate `self` | `other`."""
        return OperatorMetric("__or__", self, other)

    def __pos__(self) -> "Metric":
        """Evaluate `+self` for every element of the metric result."""
        return OperatorMetric("__abs__", self, None)

    def __pow__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Raise the metric to the power of another object."""
        return OperatorMetric("__pow__", self, other)

    def __sub__(self, other: Union[int, float, "Metric", Array]) -> "Metric":
        """Subtract two metrics, a metric and a scalar or a metric and an array."""
        return OperatorMetric("__sub__", self, other)

    def __truediv__(
        self,
        other: Union[int, float, "Metric", Array],
    ) -> "Metric":
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
            f"Expected the value of state `{name}` to be an array API object or a "
            f"list of array API-compatible objects. But got {type(value)} instead.",
        )


class OperatorMetric(Metric):
    """A metric used to apply an operator to one or two metrics.

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

    name: str = "OperatorMetric"

    def __init__(
        self,
        operator: str,
        metric_a: Union[bool, int, float, Metric, Array],
        metric_b: Optional[Union[bool, int, float, Metric, Array]],
    ) -> None:
        """Initialize the metric."""
        super().__init__()

        self._op = operator
        self.metric_a = metric_a.clone() if isinstance(metric_a, Metric) else metric_a
        self.metric_b = metric_b.clone() if isinstance(metric_b, Metric) else metric_b

    def _update_state(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of each metric."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **kwargs)

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **kwargs)

    def _compute_metric(self) -> None:
        """Not implemented and not required.

        The `compute` is overridden to call the `compute` method of `metric_a`
        and/or `metric_b` and then apply the operator.

        """

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

    def reset(self) -> None:
        """Reset the state of each metric."""
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

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

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        _op_metrics = (
            f"(\n  {self._op}(\n    {self.metric_a!r},\n    {self.metric_b!r}\n  )\n)"  # noqa: E501
        )
        return self.__class__.__name__ + _op_metrics
