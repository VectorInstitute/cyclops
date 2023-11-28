"""Utility functions for the wrappers."""

import inspect
import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

import numpy as np
import numpy.typing as npt
from datasets import Dataset
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
    from torch import Tensor as TorchTensor
    from torch import device as torch_device
    from torch import nn
    from torch.nn.utils.rnn import PackedSequence
else:
    torch = import_optional_module("torch", error="warn")
    nn = import_optional_module("torch.nn", error="warn")
    PackedSequence = import_optional_module(
        "torch.nn.utils.rnn",
        attribute="PackedSequence",
        error="warn",
    )
    torch_device = import_optional_module(
        "torch",
        attribute="device",
        error="warn",
    )
    TorchTensor = import_optional_module(
        "torch",
        attribute="Tensor",
        error="warn",
    )


def to_tensor(
    X,
    device: Union[str, torch_device] = "cpu",
    concatenate_features: bool = True,
) -> Union[TorchTensor, Sequence, Mapping]:
    """Convert the input to a torch tensor.

    Parameters
    ----------
    X : Union[torch.Tensor, numpy.ndarray, Sequence, Mapping]
        The input to convert to a tensor.
    device : str or torch.device, default="cpu"
        The device to move the tensor to.

    Returns
    -------
    torch.Tensor or Sequence or Mapping
        The converted tensor.

    Raises
    ------
    ValueError
        If ``X`` is not a numpy array, torch tensor, dictionary, list, or tuple.

    """
    if isinstance(X, (TorchTensor, PackedSequence)):
        return X.to(device)
    if np.isscalar(X):
        return torch.as_tensor(X, device=device)
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).to(device)
    if isinstance(X, Sequence):
        if concatenate_features:
            X = [
                to_tensor(x, device=device, concatenate_features=concatenate_features)
                for x in X
            ]
        else:
            X = torch.as_tensor(X, device=device)
        return X
    if isinstance(X, Mapping):
        return {
            k: to_tensor(v, device=device, concatenate_features=concatenate_features)
            for k, v in X.items()
        }
    raise ValueError(
        "Cannot convert to tensor. `X` must be a numpy array, torch tensor,"
        f" dictionary, list, or tuple. Got {type(X)} instead.",
    )


def to_numpy(X) -> Union[npt.NDArray[Any], Sequence, Mapping]:
    """Convert the input to a numpy array.

    Parameters
    ----------
    X : torch.Tensor or numpy.ndarray or Sequence or Mapping
        The input to convert to a numpy array.

    Returns
    -------
    numpy.ndarray or Sequence or Mapping
        The converted numpy array.

    Raises
    ------
    ValueError
        If ``X`` is not a numpy array, torch tensor, dictionary, list, or tuple.

    """
    if isinstance(X, np.ndarray):
        return X
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
        f" dictionary, list, or tuple. Got {type(X)} instead.",
    )


def check_is_fitted(estimator=None, attributes=None, msg=None, all_or_any=all) -> None:
    """Check if the estimator is fitted.

    This is a wrapper around sklearn.utils.validation.check_is_fitted that
    raises a NotFittedError with a more informative message.

    Parameters
    ----------
    estimator : object, default=None
        Estimator instance to check the state of.
    attributes : str or list of str, default=None
        The attributes to check if the estimator is fitted.
    msg : str, default=None
        Error message to raise if the estimator is not fitted.
    all_or_any : callable, default=all
        Function to check if all or any of the attributes are present.

    Raises
    ------
    NotFittedError
        If the model has not been fitted.

    """
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'initialize' or 'fit'"
            " with appropriate arguments before using this method."
        )
    _check_is_fitted(
        estimator=estimator,
        attributes=attributes,
        msg=msg,
        all_or_any=all_or_any,
    )


def _get_param_names(cls):
    """Get parameter names for the wrapper.

    This method is similar to ``BaseEstimator._get_param_names``, but
    is implemented as a static method to avoid the need to instantiate
    the wrapper.

    Parameters
    ----------
    cls : class
        The class to get the parameter names for.

    Returns
    -------
    list
        The parameter names.

    """
    # get the constructor or the original constructor before
    # deprecation wrapping if any
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    if init is object.__init__:  # No explicit constructor to introspect
        return []

    # introspect the constructor arguments to find the model parameter
    # to represent
    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    parameters = [
        param
        for param in init_signature.parameters.values()
        if param.name != "self" and param.kind != param.VAR_KEYWORD
    ]

    for param in parameters:
        if param.kind == param.VAR_POSITIONAL:
            raise RuntimeError(
                "Model wrappers should always specify their parameters in the signature"
                f" of their __init__ (no varargs). {cls} with constructor"
                f" {init_signature} doesn't follow this convention.",
            )
    # Extract and sort argument names excluding 'self'
    return sorted([param.name for param in parameters])


def get_params(cls) -> dict:
    """Get parameters for the wrapper.

    This method is similar to ``BaseEstimator.get_params``, but is
    implemented as a static method to avoid the need to instantiate the
    wrapper. It also does not support the ``deep`` parameter.

    Returns
    -------
    dict
        Parameter names mapped to their values.

    """
    out = {}
    for key in _get_param_names(cls):
        value = getattr(cls, key, None)
        out[key] = value
    return out


def set_params(cls, **params):
    """Set the parameters of the wrapper.

    This method is similar to ``BaseEstimator.set_params``, but is
    implemented as a static method to avoid the need to instantiate the
    wrapper. It also does not support the ``deep`` parameter.

    Parameters
    ----------
    cls : class
        The class to set the parameters for.
    **params : dict, optional
        Parameters to set.

    Returns
    -------
    cls

    """
    if not params:
        # Simple optimization to gain speed (inspect is slow)
        return cls
    valid_params = get_params(cls)

    nested_params = defaultdict(dict)  # grouped by prefix
    for key, value in params.items():
        key, delim, sub_key = key.partition("__")  # noqa: PLW2901
        if key not in valid_params:
            local_valid_params = _get_param_names(cls)
            raise ValueError(
                f"Invalid parameter {key!r} for wrapper {cls}. "
                f"Valid parameters are: {local_valid_params!r}.",
            )

        if delim:
            nested_params[key][sub_key] = value
        else:
            setattr(cls, key, value)
            valid_params[key] = value

    for key, sub_params in nested_params.items():
        valid_params[key].set_params(**sub_params)

    return cls


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set a random seed for python, numpy and PyTorch globally.

    Parameters
    ----------
    seed: int
        Value of random seed to set.
    deterministic: bool, default=False
        Turn on CuDNN deterministic settings. This will slow down training.

    """
    if seed is not None and isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # does nothing if no GPU
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(mode=deterministic)
        if not deterministic and torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True


class DatasetColumn(list):
    """Helper class to avoid loading a dataset column into memory when accessing it.

    Code taken from Hugging Face Evaluator (https://huggingface.co/docs/evaluate/index).

    """

    def __init__(self, dataset: Dataset, key: str) -> None:
        """Initialize a new dataset column.

        Parameters
        ----------
        dataset : Dataset
            Hugging Face Dataset
        key : str
            Column name

        """
        super().__init__()
        self.dataset = dataset
        self.key = key

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, i):
        """Get the `i`th item in the specified column."""
        return self.dataset[i][self.key]

    def __iter__(self):
        """Iterate over the column in the dataset."""
        return (self.dataset[i][self.key] for i in range(len(self)))

    def __all__(self):
        """Get the whole column."""
        return self.dataset[self.key]
