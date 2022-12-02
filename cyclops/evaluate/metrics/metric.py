"""Base abstract class for all metrics."""

import functools
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import numpy as np

from cyclops.evaluate.metrics.utils import (
    _apply_function_recursively,
    _get_value_if_singleton_array,
)


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

    def __call__(self, *args, **kwargs):
        """Update the global metric state and compute the metric for a batch."""
        # global accumulation
        self.update_state(*args, **kwargs)
        _update_count = self._update_count
        cache = {attr: getattr(self, attr) for attr in self._defaults}

        # batch computation
        self.reset_state()
        self.update_state(*args, **kwargs)
        batch_result = self.compute()

        # restore global state
        for attr, value in cache.items():
            setattr(self, attr, value)
        self._update_count = _update_count
        self._computed = None

        return batch_result

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

            update(*args, **kwargs)

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
