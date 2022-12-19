"""Base abstract class for all metrics."""

import functools
import inspect
import logging
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

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

    def __init__(self):
        self.update_state: Callable = self._wrap_update(
            self.update_state
        )  # type: ignore # pylint: disable=method-hidden
        self.compute: Callable = self._wrap_compute(
            self.compute
        )  # type: ignore # pylint: disable=method-hidden
        self._update_count: int = 0
        self._computed: Any = None
        self._defaults: Dict[str, Union[List, np.ndarray]] = {}

    def __init_subclass__(
        cls, registry_key: str = None, force_register: bool = False, **kwargs
    ):
        """Register the subclass in the registry."""
        super().__init_subclass__(**kwargs)

        # check that the subclass has implemented the abstract methods
        if (
            not (
                cls.update_state is Metric.update_state or cls.compute is Metric.compute
            )
            or force_register
        ):
            if registry_key is None:
                LOGGER.warning(
                    "Metric subclass %s has not defined a name. "
                    "It will not be registered in the metric registry.",
                    cls.__name__,
                )
            else:
                if registry_key in _METRIC_REGISTRY:
                    raise ValueError(
                        f"Metric with name {registry_key} is already registered."
                    )
                _METRIC_REGISTRY[registry_key] = cls

    def add_state(self, name: str, default: Union[List, np.ndarray]) -> None:
        """Add a state variable to the metric.

        Parameters
        ----------
        name: str
            The name of the state variable.
        default: Union[List, numpy.ndarray]
            The default value of the state variable.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If the state variable already exists.
        ValueError
            If the state variable is not a numpy.ndarray or an empty list.

        """
        if hasattr(self, name):
            raise AttributeError(f"Attribute {name} already exists.")

        if not isinstance(default, (np.ndarray, list)) or (
            isinstance(default, list) and default
        ):
            raise ValueError(
                "state variable must be a numpy.ndarray or any empty list"
                " (where numpy arrays can be appended)"
            )

        if isinstance(default, np.ndarray):
            default = np.ascontiguousarray(default)

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)

    def reset_state(self) -> None:
        """Reset the metric to its initial state.

        Returns
        -------
        None

        """
        self._update_count = 0
        self._computed = None

        for attr, default in self._defaults.items():
            if isinstance(default, np.ndarray):
                setattr(self, attr, default.copy())
            else:
                setattr(self, attr, [])

    @abstractmethod
    def update_state(  # pylint: disable=method-hidden
        self, *args: Any, **kwargs: Any
    ) -> None:
        """Update the state of the metric."""

    @abstractmethod
    def compute(self) -> Any:  # pylint: disable=method-hidden
        """Compute the final value of the metric from the state variables."""

    def _wrap_update(self, update: Callable) -> Callable:
        """Manage the internal attributes before calling the update method.

        Sets the ``_computed`` attribute to None and increments the ``_update_count``
        attribute before calling the custom update method.

        Parameters
        ----------
        update: Callable
            The update method of the metric.

        Returns
        -------
        wrapped_func: Callable
            The wrapped update method.

        """

        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1

            return update(*args, **kwargs)

        return wrapped_func

    def _wrap_compute(self, compute: Callable) -> Callable:
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
        UserWarning
            If the ``compute`` method has not been called at least once.

        """

        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if self._update_count == 0:
                warnings.warn(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                    UserWarning,
                )

            # return cached value
            if self._computed is not None:
                return self._computed

            value = compute(*args, **kwargs)
            self._computed = _apply_function_recursively(
                value, _get_value_if_singleton_array
            )

            return self._computed

        return wrapped_func

    def __call__(self, *args, **kwargs):
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


def create_metric(metric_name: str, **kwargs: Optional[Dict[str, Any]]) -> Metric:
    """Create a metric instance from a name.

    Parameters
    ----------
    metric_name: str
        The name of the metric.
    **kwargs: Optional[Dict[str, Any]]
        The keyword arguments to pass to the metric constructor.

    Returns
    -------
    metric: Metric
        The metric instance.

    """
    metric_class = _METRIC_REGISTRY.get(metric_name, None)
    if metric_class is None:
        similar_keys_list: List[str] = get_close_matches(
            metric_name, _METRIC_REGISTRY.keys(), n=5
        )
        similar_keys: str = ", ".join(similar_keys_list)
        similar_keys = (
            f" Did you mean one of: {similar_keys}?"
            if similar_keys
            else " It may not be registered."
        )
        raise ValueError(f"Metric {metric_name} not found.{similar_keys}")

    metric = metric_class(**kwargs)

    return metric


class MetricCollection(Metric):
    """A collection of metrics.

    Provides a convenient way to compute multiple metrics at once. It groups
    metrics that have similar state variables and only updates the state variables
    once per group, reducing the amount of computation required.

    Parameters
    ----------
    metrics: List[Metric]
        The list of metrics to collect.

    """

    def __init__(self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]]):
        super().__init__()

        self._validate_input(metrics)

        if isinstance(metrics, Metric):
            self._metrics = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, Sequence):
            self._metrics = {metric.__class__.__name__: metric for metric in metrics}
        else:
            self._metrics = metrics

        self.metric_groups = self._get_metric_groups()

    def add_state(self, *args: Any, **kwargs: Any) -> None:
        """Add state variables to the metric.

        Not implemented for ``MetricCollection``.

        """
        raise NotImplementedError(
            "The ``add_state`` method is not supported for the ``MetricCollection``"
            " class."
        )

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
        for metric_names in self.metric_groups.values():
            base_metric = self._metrics[metric_names[0]]
            base_metric.update_state(*args, **kwargs)

            for metric_name in metric_names[1:]:
                metric = self._metrics[metric_name]
                for attr in metric._defaults.keys():  # pylint: disable=protected-access
                    setattr(metric, attr, getattr(base_metric, attr))

                # pylint: disable=protected-access
                metric._computed = base_metric._computed
                metric._update_count = base_metric._update_count

    def compute(self) -> Dict[str, Any]:
        """Compute the metrics in the collection."""
        return {name: metric.compute() for name, metric in self._metrics.items()}

    def reset_state(self) -> None:
        """Reset the state of all metrics in the collection."""
        for metric in self._metrics.values():
            metric.reset_state()

    def _validate_input(
        self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]]
    ):
        """Validate the input to the constructor.

        Parameters
        ----------
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]]
            The input to the constructor.

        Raises
        ------
        TypeError
            If the input is not a metric, sequence of metrics, or dictionary of metrics.

        """
        if isinstance(metrics, Metric):
            if not isinstance(metrics, Metric):
                raise TypeError(
                    f"Metric {metrics.__class__.__name__} is not a subclass of Metric."
                )
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, Metric):
                    raise TypeError(
                        f"Metric {metric.__class__.__name__} is not an instance"
                        " of `Metric`."
                    )
                self._has_same_update_state_parameters_check(metrics[0], metric)
                self._has_same_task_type_check(metrics[0], metric)
        elif isinstance(metrics, Dict):
            first_metric = next(iter(metrics.values()))
            for metric in metrics.values():
                if not isinstance(metric, Metric):
                    raise TypeError(
                        f"Metric {metric.__class__.__name__} is not an instance of"
                        " `Metric`."
                    )
                self._has_same_update_state_parameters_check(first_metric, metric)
                self._has_same_task_type_check(first_metric, metric)
        else:
            raise TypeError(
                f"Metrics must be a Metric, a Sequence of Metrics or a Dict of Metrics,"
                f" but got {type(metrics)}."
            )

    @staticmethod
    def _has_same_update_state_parameters_check(
        metric_a: Metric, metric_b: Metric
    ) -> None:
        """Check if two metrics have the same ``update_state`` method parameters.

        Parameters
        ----------
        metric_a: Metric
            The first metric.
        metric_b: Metric
            The second metric.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the metrics do not have the same signature for their ``update_state``
            method.

        """
        if (
            not inspect.signature(metric_a.update_state).parameters
            == inspect.signature(metric_b.update_state).parameters
        ):
            raise ValueError(
                "All metrics in the collection must have the same signature for"
                f" the `update_state` method. Metric {metric_a.__class__.__name__}"
                f" has signature {inspect.signature(metric_a.update_state)},"
                f" but metric {metric_b.__class__.__name__} has signature"
            )

    @staticmethod
    def _has_same_task_type_check(metric_a: Metric, metric_b: Metric) -> None:
        """Check if two metrics are for the same task.

        Parameters
        ----------
        metric_a: Metric
            The first metric.
        metric_b: Metric
            The second metric.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the metrics are not for the same task. If both metrics are for
            multilabel tasks, then they must have the same ``num_labels``
            parameter. If both metrics are for multiclass tasks, then they must
            have the same ``num_classes`` parameter.

        """
        metric_types = re.compile(r"(Binary|Multiclass|Multilabel)(.*)")
        metric_a_match = metric_types.match(metric_a.__class__.__name__)
        metric_b_match = metric_types.match(metric_b.__class__.__name__)

        if (
            metric_a_match is None
            or metric_b_match is None
            or (metric_a_match.group(1) != metric_b_match.group(1))
        ):
            raise ValueError(
                "All metrics in the collection must be for the same task type"
                " (binary, multiclass, or multilabel)."
            )

        if metric_a_match.group(1) == "Multiclass" and (
            metric_a.__dict__["num_classes"] != metric_b.__dict__["num_classes"]
        ):
            raise ValueError(
                "All multiclass metrics in the collection must have the same"
                f" number of classes. Metric {metric_a.__class__.__name__}"
                f" has {metric_a.__dict__['num_classes']} classes, but metric"
                f" {metric_b.__class__.__name__} has {metric_b.__dict__['num_classes']}"
                " classes."
            )
        if metric_a_match.group(1) == "Multilabel" and (
            metric_a.__dict__["num_labels"] != metric_b.__dict__["num_labels"]
        ):
            raise ValueError(
                "All multilabel metrics in the collection must have the same"
                f" number of labels. Metric {metric_a.__class__.__name__}"
                f" has {metric_a.__dict__['num_labels']} labels, but metric"
                f" {metric_b.__class__.__name__} has {metric_b.__dict__['num_labels']}"
                " labels."
            )

    def _get_metric_groups(self) -> Dict[int, List[str]]:
        """Group metrics by the state variables they use.

        Returns
        -------
        metric_groups: Dict[int, List[str]]
            A dictionary with the group id as the key and a list of metric names
            as the value.

        """
        metrics_by_state = defaultdict(list)
        for metric_name, metric in self._metrics.items():
            metrics_by_state[
                tuple(metric._defaults.keys())  # pylint: disable=protected-access
            ].append(metric_name)

        metric_groups: Dict[int, List[str]] = {}
        for i, metric_names in enumerate(metrics_by_state.values()):
            metric_groups[i] = metric_names

        return metric_groups

    def __call__(self, *args, **kwargs):
        """Update the global metric state and compute the metric for a batch."""
        # global accumulation
        self.update_state(*args, **kwargs)
        update_count = self._update_count
        cache = {
            i: {
                attr: getattr(self._metrics[metric_names[0]], attr)
                for attr in self._metrics[metric_names[0]]._defaults.keys()
            }
            for i, metric_names in self.metric_groups.items()
        }  # cache the _defaults attribute for each group of metrics

        # batch computation
        self.reset_state()
        self.update_state(*args, **kwargs)
        batch_result = self.compute()

        # restore global state
        self._update_count = update_count
        self._computed = None
        for i, metric_names in self.metric_groups.items():
            for metric_name in metric_names:
                metric = self._metrics[metric_name]
                for attr, value in cache[i].items():
                    setattr(metric, attr, value)

                metric._computed = None
                metric._update_count = self._update_count

        return batch_result
