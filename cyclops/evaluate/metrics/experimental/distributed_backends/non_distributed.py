"""A dummy object for non-distributed environments."""

from typing import Any, List

from cyclops.evaluate.metrics.experimental.distributed_backends.base import (
    DistributedBackend,
)


class NonDistributed(DistributedBackend, registry_key="non_distributed"):
    """A dummy distributed communication backend for non-distributed environments."""

    @property
    def is_initialized(self) -> bool:
        """Return `True` if the distributed environment has been initialized.

        For a non-distributed environment, it is always `False`.
        """
        return False

    @property
    def rank(self) -> int:
        """Return the rank of the current process.

        For a non-distributed environment, it is always 0
        """
        return 0

    @property
    def world_size(self) -> int:
        """Return the total number of processes.

        For a non-distributed environment, it is always 1.
        """
        return 1

    def all_gather(self, arr: Any) -> List[Any]:
        """Return the input as a list."""
        return [arr]
