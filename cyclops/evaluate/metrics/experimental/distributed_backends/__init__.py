"""Distributed backends for distributed metric computation."""

from cyclops.evaluate.metrics.experimental.distributed_backends.base import (
    _DISTRIBUTED_BACKEND_REGISTRY,
    DistributedBackend,
)
from cyclops.evaluate.metrics.experimental.distributed_backends.mpi4py import (
    MPI4Py,
)
from cyclops.evaluate.metrics.experimental.distributed_backends.non_distributed import (
    NonDistributed,
)
from cyclops.evaluate.metrics.experimental.distributed_backends.torch_distributed import (
    TorchDistributed,
)


def get_backend(name: str) -> DistributedBackend:
    """Return a registered distributed backend by name.

    Parameters
    ----------
    name : str
        Name of the distributed backend.

    Returns
    -------
    backend : DistributedBackend
        An instance of the distributed backend.

    Raises
    ------
    ValueError
        If the backend is not found in the registry.

    """
    if not isinstance(name, str):
        raise TypeError(
            f"Expected `name` to be a str, but got {type(name)}.",
        )

    name = name.lower()
    backend = _DISTRIBUTED_BACKEND_REGISTRY.get(name)
    if backend is None:
        raise ValueError(
            f"Backend `{name}` is not found. "
            f"It should be one of {list(_DISTRIBUTED_BACKEND_REGISTRY.keys())}.",
        )
    return backend()
