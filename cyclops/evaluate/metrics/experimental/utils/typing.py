"""Utilities for array-API compatibility."""
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union

import numpy.typing as npt
import torch

from cyclops.utils.optional import import_optional_module


class _ArrayAPICompliantObject(Protocol):
    """Protocol for objects that have a __array_namespace__ attribute."""

    def __array_namespace__(self, api_version: Optional[str] = None) -> Any:
        """Return an array-API-compatible namespace."""
        ...


_supported_array_types = (npt.NDArray, torch.Tensor, _ArrayAPICompliantObject)

cp = import_optional_module("cupy", error="ignore")
if cp is not None:
    _supported_array_types += (cp.ndarray,)  # type: ignore[assignment]

if TYPE_CHECKING:  # noqa: SIM108
    Array = Any
else:
    Array = Union[_supported_array_types]  # type: ignore
