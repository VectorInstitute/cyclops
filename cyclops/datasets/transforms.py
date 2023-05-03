"""Transforms for the datasets."""

from functools import partial
from typing import Tuple
from torchvision.transforms import Compose
from monai.transforms import (
    Resized,
    Lambdad,
    ToDeviced,
)
def txrv_transforms(keys: Tuple[str, ...] = ("features",), device: str = "cpu") -> Compose:
    """Transforms for the models in the TXRV library."""
    transforms = Compose(
        [
            Resized(
                keys=keys,
                spatial_size=(1, 224, 224),
                allow_missing_keys=True,
            ),
            Lambdad(
                keys=keys,
                func=lambda x: ((2 * (x / 255.0)) - 1.0) * 1024,
                allow_missing_keys=True,
            ),
            ToDeviced(keys=keys, device=device, allow_missing_keys=True),
        ],
    )
    return transforms
