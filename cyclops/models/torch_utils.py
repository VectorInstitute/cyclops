"""Utility functions for building pytorch based models."""

import inspect
from difflib import get_close_matches
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
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
    if PackedSequence is None:
        PackedSequence = type(None)


def _get_class_members(
    module,
    include_only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> dict:
    """Get class members from module.

    Parameters
    ----------
    module : module
        Module to get class members from.
    include_only : list[str], optional
        List of class names to include.
    exclude : list[str], optional
        List of class names to exclude.

    Returns
    -------
    dict
        Dictionary of class members.

    """
    if include_only is None:
        include_only = []
    if exclude is None:
        exclude = []
    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if name in include_only or (not include_only and name not in exclude)
    }


####################
# Loss catalog     #
####################
_criterion_catalog: Dict[str, torch.nn.modules.loss._Loss] = _get_class_members(
    torch.nn.modules.loss,
    include_only=["BCELoss", "BCEWithLogitsLoss"],
)

#####################
# Optimizer catalog #
#####################
_optimizer_catalog: Dict[str, torch.optim.Optimizer] = _get_class_members(
    torch.optim,
    exclude=["Optimizer"],
)

#####################
# Scheduler catalog #
#####################
_lr_scheduler_catalog: Dict[
    str,
    torch.optim.lr_scheduler._LRScheduler,
] = _get_class_members(
    torch.optim.lr_scheduler,
    exclude=["_LRScheduler", "Optimizer", "Counter", "ChainedScheduler"],
)

######################
# Activation catalog #
######################
_activation_catalog = _get_class_members(
    torch.nn.modules.activation,
    include_only=[
        "Hardtanh",
        "Sigmoid",
        "ReLU6",
        "Tanh",
        "Tanhshrink",
        "Hardshrink",
        "LeakyReLU",
        "Softshrink",
        "ReLU",
        "PReLU",
        "Softplus",
        "ELU",
        "SELU",
        "SiLU",
    ],
)
_activation_catalog.update(Id=torch.nn.Identity)


def get_module(module_type: str, module_name: str):
    """Get module.

    Parameters
    ----------
    module_type : str
        Module type.
    module_name : str
        Module name.

    Returns
    -------
    Any
        Module.

    Raises
    ------
    ValueError
        If module type is not supported.

    """
    if module_type == "criterion":
        catalog = _criterion_catalog
    elif module_type == "optimizer":
        catalog = _optimizer_catalog
    elif module_type == "lr_scheduler":
        catalog = _lr_scheduler_catalog
    elif module_type == "activation":
        catalog = _activation_catalog
    else:
        raise ValueError(f"Module type {module_type} is not supported.")

    module = catalog.get(module_name, None)
    if module is None:
        similar_keys_list: List[str] = get_close_matches(
            module_name,
            catalog.keys(),
            n=5,
        )
        similar_keys: str = ", ".join(similar_keys_list)
        similar_keys = (
            f" Did you mean one of: {similar_keys}?"
            if similar_keys
            else "It may not be in the catalog."
        )
        raise ValueError(f"Module {module_name} not found.{similar_keys}")

    return module


def get_device() -> torch.device:
    """Get device for PyTorch models.

    Returns
    -------
    torch.device
        cpu or cuda

    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(
    X,
    device: Union[str, torch.device] = "cpu",
    concatenate_features: bool = True,
) -> Union[torch.Tensor, Sequence, Mapping]:
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
    if isinstance(X, (torch.Tensor, PackedSequence)):
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


class DefaultCriterion(nn.Module):
    """Default criterion for the wrapper.

    Returns the mean value of the model logits.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        """Forward pass of the criterion."""
        return preds.mean()


class LossMeter:
    """Loss meter for PyTorch models."""

    def __init__(self, name: str) -> None:
        """Initialize loss meter.

        Parameters
        ----------
        name : str
            Loss name. Used for logging.

        """
        self.name = name
        self.losses: List[float] = []

    def reset(self) -> None:
        """Reset the list of losses."""
        self.losses.clear()

    def add(self, val: float) -> None:
        """Add to the list of losses.

        Parameters
        ----------
        val : float
            Loss value.

        """
        self.losses.append(val)

    def mean(self) -> float:
        """Get the mean of the loss values in the list.

        Returns
        -------
        float
            Mean values.

        """
        if not self.losses:
            return 0
        return np.mean(self.losses)

    def pop(self) -> float:
        """Get the last element of the list.

        Returns
        -------
        float
            Loss value.

        """
        return self.losses[-1]

    def sum(self) -> float:  # noqa: A003
        """Get the summation of all loss values.

        Returns
        -------
        float
            Sum value.

        """
        return sum(self.losses)
