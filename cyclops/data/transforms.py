"""Transforms for the datasets."""

from typing import TYPE_CHECKING, Any, Callable, Tuple

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from torchvision.transforms import Lambda, Resize
else:
    Lambda = import_optional_module(
        "torchvision.transforms",
        attribute="Lambda",
        error="warn",
    )
    Resize = import_optional_module(
        "torchvision.transforms",
        attribute="Resize",
        error="warn",
    )


# generic dictionary-based wrapper for any transform
class Dictd:
    """Generic dictionary-based wrapper for any transform."""

    def __init__(
        self,
        transform: Callable[..., Any],
        keys: Tuple[str, ...],
        allow_missing_keys: bool = False,
    ):
        self.transform = transform
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Any) -> Any:
        """Apply the transform to the data."""
        for key in self.keys:
            if self.allow_missing_keys and key not in data:
                continue
            data[key] = self.transform(data[key])
        return data

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return (
            f"{self.__class__.__name__}(transform={self.transform}, "
            f"keys={self.keys}, allow_missing_keys={self.allow_missing_keys})"
        )


# dictionary-based wrapper of Lambda transform using Dictd
class Lambdad:
    """Dictionary-based wrapper of Lambda transform using Dictd."""

    def __init__(
        self,
        func: Callable[..., Any],
        keys: Tuple[str, ...],
        allow_missing_keys: bool = False,
    ):
        self.transform = Dictd(
            transform=Lambda(func),
            keys=keys,
            allow_missing_keys=allow_missing_keys,
        )

    def __call__(self, data: Any) -> Any:
        """Apply the transform to the data."""
        return self.transform(data)

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return f"{self.__class__.__name__}(keys={self.transform.keys}, allow_missing_keys={self.transform.allow_missing_keys})"


# dictionary-based wrapper of Resize transform using Dictd
class Resized:
    """Dictionary-based wrapper of Resize transform using Dictd."""

    def __init__(
        self,
        spatial_size: Tuple[int, int],
        keys: Tuple[str, ...],
        allow_missing_keys: bool = False,
    ):
        self.transform = Dictd(
            transform=Resize(size=spatial_size),
            keys=keys,
            allow_missing_keys=allow_missing_keys,
        )

    def __call__(self, data: Any) -> Any:
        """Apply the transform to the data."""
        return self.transform(data)

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return f"{self.__class__.__name__}(keys={self.transform.keys}, allow_missing_keys={self.transform.allow_missing_keys})"
