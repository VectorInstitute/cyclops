"""Tasks utility functions."""
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import PIL
import torch
from torchvision.transforms import PILToTensor


def apply_image_transforms(examples: Dict[str, List], transforms: callable) -> dict:
    """Apply transforms to examples.

    Used for applying image transformations to examples for chest X-ray classification.

    """
    # examples is a dict of lists; convert to list of dicts.
    # doing a conversion from PIL to tensor is necessary here when working
    # with the Image feature type.
    value_len = len(list(examples.values())[0])
    examples = [
        {
            k: PILToTensor()(v[i]) if isinstance(v[i], PIL.Image.Image) else v[i]
            for k, v in examples.items()
        }
        for i in range(value_len)
    ]

    # apply the transforms to each example
    examples = [transforms(example) for example in examples]

    # convert back to a dict of lists
    examples = {k: [d[k] for d in examples] for k in examples[0]}

    return examples


def to_numpy(X) -> np.ndarray:
    """Convert input to a numpy array.

    Parameters
    ----------
    X : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, Mapping, Sequence]
        Input data.

    Returns
    -------
    np.ndarray
        Output numpy array.

    Raises
    ------
    ValueError
        Input type is not supported.

    """
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()

    if isinstance(X, torch.Tensor):
        if X.requires_grad:
            X = X.detach()

        if X.is_cuda:
            X = X.cpu()
        return X.numpy()

    if np.isscalar(X):
        return np.array(X)

    if isinstance(X, Sequence):
        return type(X)(to_numpy(x) for x in X)
    if isinstance(X, Mapping):
        return {k: to_numpy(v) for k, v in X.items()}

    raise ValueError(
        "Cannot convert to numpy array. `X` must be a numpy array, torch tensor,"
        f" dictionary, list, tuple, pandas dataframe or pandas series. \
        Got {type(X)} instead."
    )
