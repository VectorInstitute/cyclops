"""Utilities for array-API compatibility."""

import builtins
from enum import IntEnum
from types import ModuleType
from typing import Any, Optional, Protocol, Tuple, TypeVar, Union

from typing_extensions import TypeAlias


PyCapsule = Any
Device = Any
Dtype = Any

T = TypeVar("T")


class SupportsArrayAPI(Protocol[T]):
    """Protocol for objects that implement the array API standard specifications.

    This protocol mostly follows the v2021.12 specification.

    """

    def __abs__(self: T, /) -> T:
        """Perform the operation __abs__."""
        ...

    def __add__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __add__."""
        ...

    def __and__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __and__."""
        ...

    def __array_namespace__(
        self: T,
        /,
        *,
        api_version: Optional[str] = None,
    ) -> ModuleType:
        """Return the array API namespace."""
        ...

    def __bool__(self: T, /) -> bool:
        """Perform the operation __bool__."""
        ...

    def __dlpack__(self: T, /, *, stream: None = None) -> PyCapsule:
        """Perform the operation __dlpack__."""
        ...

    def __dlpack_device__(self: T, /) -> Tuple[IntEnum, int]:
        """Perform the operation __dlpack_device__."""
        ...

    def __eq__(  # type: ignore[override]
        self: T,
        other: Union[int, float, bool, T],  # type: ignore[override]
        /,
    ) -> T:
        """Perform the operation __eq__."""
        ...

    def __float__(self: T, /) -> float:
        """Perform the operation __float__."""
        ...

    def __floordiv__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __floordiv__."""
        ...

    def __ge__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __ge__."""
        ...

    def __getitem__(
        self: T,
        key: Union[
            int,
            slice,
            "builtins.ellipsis",
            Tuple[Union[int, slice, "builtins.ellipsis"], ...],
            T,
        ],
        /,
    ) -> T:
        """Perform the operation __getitem__."""
        ...

    def __gt__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __gt__."""
        ...

    def __int__(self: T, /) -> int:
        """Perform the operation __int__."""
        ...

    def __index__(self: T, /) -> int:
        """Perform the operation __index__."""
        ...

    def __invert__(self: T, /) -> T:
        """Perform the operation __invert__."""
        ...

    def __le__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __le__."""
        ...

    def __lshift__(self: T, other: Union[int, T], /) -> T:
        """Perform the operation __lshift__."""
        ...

    def __lt__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __lt__."""
        ...

    def __matmul__(self: T, other: T, /) -> T:
        """Perform the operation __matmul__."""
        ...

    def __mod__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __mod__."""
        ...

    def __mul__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __mul__."""
        ...

    def __ne__(  # type: ignore[override]
        self: T,
        other: Union[int, float, bool, T],  # type: ignore[override]
        /,
    ) -> T:
        """Perform the operation __ne__."""
        ...

    def __neg__(self: T, /) -> T:
        """Perform the operation __neg__."""
        ...

    def __or__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __or__."""
        ...

    def __pos__(self: T, /) -> T:
        """Perform the operation __pos__."""
        ...

    def __pow__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __pow__."""
        ...

    def __rshift__(self: T, other: Union[int, T], /) -> T:
        """Perform the operation __rshift__."""
        ...

    def __setitem__(
        self,
        key: Union[
            int,
            slice,
            "builtins.ellipsis",
            Tuple[Union[int, slice, "builtins.ellipsis"], ...],
            T,
        ],
        value: Union[int, float, bool, T],
        /,
    ) -> None:
        """Perform the operation __setitem__."""
        ...

    def __sub__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __sub__."""
        ...

    # PEP 484 requires int to be a subtype of float, but __truediv__ should
    # not accept int.
    def __truediv__(self: T, other: Union[float, T], /) -> T:
        """Perform the operation __truediv__."""
        ...

    def __xor__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __xor__."""
        ...

    def __iadd__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __iadd__."""
        ...

    def __radd__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __radd__."""
        ...

    def __iand__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __iand__."""
        ...

    def __rand__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __rand__."""
        ...

    def __ifloordiv__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __ifloordiv__."""
        ...

    def __rfloordiv__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __rfloordiv__."""
        ...

    def __ilshift__(self: T, other: Union[int, T], /) -> T:
        """Perform the operation __ilshift__."""
        ...

    def __rlshift__(self: T, other: Union[int, T], /) -> T:
        """Perform the operation __rlshift__."""
        ...

    def __imatmul__(self: T, other: T, /) -> T:
        """Perform the operation __imatmul__."""
        ...

    def __rmatmul__(self: T, other: T, /) -> T:
        """Perform the operation __rmatmul__."""
        ...

    def __imod__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __imod__."""
        ...

    def __rmod__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __rmod__."""
        ...

    def __imul__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __imul__."""
        ...

    def __rmul__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __rmul__."""
        ...

    def __ior__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __ior__."""
        ...

    def __ror__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __ror__."""
        ...

    def __ipow__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __ipow__."""
        ...

    def __rpow__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __rpow__."""
        ...

    def __irshift__(self: T, other: Union[int, T], /) -> T:
        """Perform the operation __irshift__."""
        ...

    def __rrshift__(self: T, other: Union[int, T], /) -> T:
        """Perform the operation __rrshift__."""
        ...

    def __isub__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __isub__."""
        ...

    def __rsub__(self: T, other: Union[int, float, T], /) -> T:
        """Perform the operation __rsub__."""
        ...

    def __itruediv__(self: T, other: Union[float, T], /) -> T:
        """Perform the operation __itruediv__."""
        ...

    def __rtruediv__(self: T, other: Union[float, T], /) -> T:
        """Perform the operation __rtruediv__."""
        ...

    def __ixor__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __ixor__."""
        ...

    def __rxor__(self: T, other: Union[int, bool, T], /) -> T:
        """Perform the operation __rxor__."""
        ...

    def to_device(self: T, device: Device, /, stream: None = None) -> T:
        """Move the array to the specified device."""
        ...

    @property
    def dtype(self) -> Dtype:
        """Return the data type of the array."""
        ...

    @property
    def device(self) -> Device:
        """Return the device on which the array is stored."""
        ...

    @property
    def mT(self) -> T:  # noqa: N802
        """Return the matrix transpose of the array."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the array."""
        ...

    @property
    def size(self) -> int:
        """Return the number of elements in the array."""
        ...

    @property
    def T(self) -> T:  # noqa: N802
        """Return the transpose of the array."""
        ...


Array: TypeAlias = SupportsArrayAPI[Any]
