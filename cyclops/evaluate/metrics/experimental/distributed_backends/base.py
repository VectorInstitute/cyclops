"""Distributed backend interface."""
import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, List, Optional

from cyclops.evaluate.metrics.experimental.utils.typing import Array
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
    ):
        """Register the distributed backend."""
        super().__init_subclass__(**kwargs)

        if (  # subclass has not implemented abstract methods/properties
            cls.all_gather is not DistributedBackend.all_gather
            and cls.is_initialized is not DistributedBackend.is_initialized
            and cls.rank is not DistributedBackend.rank
            and cls.world_size is not DistributedBackend.world_size
        ) and cls.__name__ != "DistributedBackend":
            if registry_key is None:
                registry_key = cls.__name__
            elif registry_key in _DISTRIBUTED_BACKEND_REGISTRY:
                LOGGER.warning(
                    "The given distributed backend %s has already been registered. "
                    "It will be overwritten by %s.",
                    registry_key,
                    cls.__name__,
                )
            _DISTRIBUTED_BACKEND_REGISTRY[registry_key] = cls
        else:
            LOGGER.warning(
                "The distributed backend %s is not registered because it does not "
                "implement any abstract methods/properties.",
                cls.__name__,
            )

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
