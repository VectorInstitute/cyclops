"""mpi4py backend for synchronizing array-API-compatible objects."""

# mypy: disable-error-code="no-any-return,arg-type"
import os
from typing import TYPE_CHECKING, List

import array_api_compat as apc

from cyclops.evaluate.metrics.experimental.distributed_backends.base import (
    DistributedBackend,
)
from cyclops.evaluate.metrics.experimental.utils.ops import flatten
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from mpi4py import MPI
else:
    MPI = import_optional_module("mpi4py.MPI", error="warn")


class MPI4Py(DistributedBackend, registry_key="mpi4py"):
    """A distributed communication backend for mpi4py."""

    def __init__(self) -> None:
        """Initialize the MPI4Py backend."""
        super().__init__()
        if MPI is None:
            raise ImportError(
                f"For availability of {self.__class__.__name__},"
                " please install mpi4py first.",
            )

    @property
    def is_initialized(self) -> bool:
        """Return `True` if the distributed environment has been initialized."""
        return "OMPI_COMM_WORLD_SIZE" in os.environ

    @property
    def rank(self) -> int:
        """Return the rank of the current process."""
        comm = MPI.COMM_WORLD
        return comm.Get_rank()

    @property
    def world_size(self) -> int:
        """Return the total number of processes."""
        comm = MPI.COMM_WORLD
        return comm.Get_size()

    def all_gather(self, arr: Array) -> List[Array]:
        """Gather Arrays from current process and return as a list.

        Parameters
        ----------
        arr : Array
            Any array-API-compatible object.

        Returns
        -------
        List[Array]
            A list of the gathered array-API-compatible objects.
        """
        try:
            xp = apc.array_namespace(arr)
        except TypeError as e:
            raise TypeError(
                "The given array is not array-API-compatible. "
                "Please use array-API-compatible objects.",
            ) from e

        comm = MPI.COMM_WORLD

        # gather the shape and size of each array
        local_shape = arr.shape
        local_size = apc.size(arr)
        all_shapes = comm.allgather(local_shape)
        all_sizes = comm.allgather(local_size)

        # prepare displacements for `Allgatherv``
        displacements = [0]
        for shape in all_shapes[:-1]:
            shape_arr = xp.asarray(shape, dtype=xp.int32)
            displacements.append(displacements[-1] + int(xp.prod(shape_arr)))

        # allocate memory for gathered data based on total size
        total_size = sum(
            [int(xp.prod(xp.asarray(shape, dtype=xp.int32))) for shape in all_shapes],
        )
        gathered_data = xp.empty(total_size, dtype=arr.dtype)

        # gather data from all processes to all processes, accounting for uneven shapes
        comm.Allgatherv(
            flatten(arr),
            [gathered_data, (all_sizes, displacements)],
        )

        # reshape gathered data back to original shape
        reshaped_data = []
        for shape in all_shapes:
            shape_arr = xp.asarray(shape, dtype=xp.int32)
            reshaped_data.append(
                xp.reshape(gathered_data[: xp.prod(shape_arr)], shape=shape),
            )
            gathered_data = gathered_data[xp.prod(shape_arr) :]

        return reshaped_data
