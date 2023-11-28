"""Data classes."""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
else:
    torch = import_optional_module("torch", error="warn")
    Tensor = import_optional_module(
        "torch",
        attribute="Tensor",
        error="warn",
    )
    Dataset = import_optional_module(
        "torch.utils.data",
        attribute="Dataset",
        error="warn",
    )


class PTDataset(Dataset):
    """General dataset wrapper that can be used in conjunction with PyTorch DataLoader.

    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor]
      Everything pertaining to the input data.

    y : Union[np.ndarray, torch.Tensor] or None (default=None)
      Everything pertaining to the target, if there is anything.

    """

    def __init__(
        self,
        X: Union[np.ndarray, Tensor],
        y: Optional[Union[np.ndarray, Tensor]] = None,
    ) -> None:
        self.X = X
        self.y = y

        len_X = len(X)  # noqa: N806
        if y is not None:
            len_y = len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._len

    def transform(self, X, y):
        """Transform the data."""
        y = torch.Tensor([0]) if y is None else y
        return (X, y)

    def __getitem__(self, idx):
        """Return the data at index idx."""
        X = self.X[idx]
        y = self.y[idx] if self.y is not None else None

        if y is not None:
            return X, y
        return X
