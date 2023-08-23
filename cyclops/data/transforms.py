"""Transforms for the datasets."""

from typing import Tuple

from monai.transforms import Lambdad, Resized, ToDeviced  # type: ignore
from torchvision.transforms import Compose


def txrv_transforms(
    keys: Tuple[str, ...] = ("features",),
    device: str = "cpu",
) -> Compose:
    """Set of transforms for the models in the TXRV library."""
    return Compose(
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
