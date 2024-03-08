"""Distributed backend interface."""

import inspect
import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Optional

from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.utils.log import setup_logging


_DISTRIBUTED_BACKEND_REGISTRY = {}
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


class DistributedBackend(ABC):
    """Abstract base class for implementing distributed communication backends.

    Parameters
    ----------
    registry_key : str, optional
        The key used to register the distributed backend. If not given, the class
        name will be used as the key.

    """

    def __init_subclass__(
        cls,
        registry_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Register the distributed backend."""
        super().__init_subclass__(**kwargs)

        if registry_key is not None and not isinstance(registry_key, str):
            raise TypeError(
                "Expected `registry_key` to be `None` or a string, but got "
                f"{type(registry_key)}.",
            )

        if not inspect.isabstract(cls) and registry_key is not None:
            _DISTRIBUTED_BACKEND_REGISTRY[registry_key] = cls

    @abstractproperty
    def is_initialized(self) -> bool:
        """Return `True` if the distributed environment has been initialized."""

    @abstractproperty
    def rank(self) -> int:
        """Return the rank of the current process."""

    @abstractproperty
    def world_size(self) -> int:
        """Return the total number of processes."""

    @abstractmethod
    def all_gather(self, arr: Array) -> List[Array]:
        """Gather Array object from all processes and return as a list.

        NOTE: This method must handle uneven array shapes.

        Parameters
        ----------
        arr : Array
           The array to be gathered.

        Returns
        -------
        List[Array]
            A list of data gathered from all processes.

        """
