"""`torch.distributed` backend for synchronizing `torch.Tensor` objects."""

from typing import TYPE_CHECKING, List

from cyclops.evaluate.metrics.experimental.distributed_backends.base import (
    DistributedBackend,
)
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
    import torch.distributed as torch_dist
    from torch import Tensor
else:
    torch = import_optional_module("torch", error="warn")
    Tensor = import_optional_module("torch", attribute="Tensor", error="warn")
    torch_dist = import_optional_module("torch.distributed", error="warn")


class TorchDistributed(DistributedBackend, registry_key="torch_distributed"):
    """A distributed communication backend for torch.distributed."""

    def __init__(self) -> None:
        """Initialize the object."""
        super().__init__()
        if torch is None:
            raise ImportError(
                f"For availability of `{self.__class__.__name__}`,"
                " please install `torch` first.",
            )
        if not torch_dist.is_available():
            raise RuntimeError(
                f"For availability of `{self.__class__.__name__}`,"
                " make sure `torch.distributed` is available.",
            )

    @property
    def is_initialized(self) -> bool:
        """Return `True` if the distributed environment has been initialized."""
        return torch_dist.is_initialized()

    @property
    def rank(self) -> int:
        """Return the rank of the current process group."""
        return torch_dist.get_rank()

    @property
    def world_size(self) -> int:
        """Return the world size of the current process group."""
        return torch_dist.get_world_size()

    def _simple_all_gather(self, data: Tensor) -> List[Tensor]:
        """Gather tensors of the same shape from all processes."""
        gathered_data = [torch.zeros_like(data) for _ in range(self.world_size)]
        torch_dist.all_gather(gathered_data, data)  # type: ignore[no-untyped-call]
        return gathered_data

    def all_gather(self, data: Tensor) -> List[Tensor]:  # type: ignore[override]
        """Gather Arrays from current process and return as a list.

        Parameters
        ----------
        arr : torch.Tensor
            `torch.Tensor` object to be gathered.

        Returns
        -------
        List[Array]
            A list of the gathered `torch.Tensor` objects.
        """
        data = data.contiguous()

        if data.ndim == 0:
            return self._simple_all_gather(data)

        # gather sizes of all tensors
        local_size = torch.tensor(data.shape, device=data.device)
        local_sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        torch_dist.all_gather(local_sizes, local_size)  # type: ignore[no-untyped-call]
        max_size = torch.stack(local_sizes).max(dim=0).values
        all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

        # if shapes are all the same, do a simple gather
        if all_sizes_equal:
            return self._simple_all_gather(data)

        # if not, pad each local tensor to maximum size, gather and then truncate
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        data_padded = torch.nn.functional.pad(data, pad_dims)
        gathered_data = [torch.zeros_like(data_padded) for _ in range(self.world_size)]
        torch_dist.all_gather(gathered_data, data_padded)  # type: ignore[no-untyped-call]
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_data[idx] = gathered_data[idx][slice_param]
        return gathered_data


if __name__ == "__main__":  # prevent execution of module on import
    pass
