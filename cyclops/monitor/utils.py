"""Utilities for the drift detector module."""

import inspect
from typing import TYPE_CHECKING, Any, Dict, Optional

from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = import_optional_module("torch", error="warn")
    nn = import_optional_module("torch.nn", error="warn")


def get_args(obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Get valid arguments from kwargs to pass to object.

    Parameters
    ----------
    obj
        object to get arguments from.
    kwargs
        Dictionary of arguments to pass to object.

    Returns
    -------
    args
        Dictionary of valid arguments to pass to class object.

    """
    args = {}
    for key, value in kwargs.items():
        if (inspect.isclass(obj) and key in inspect.signature(obj).parameters) or (
            (inspect.ismethod(obj) or inspect.isfunction(obj))
            and key in inspect.getfullargspec(obj).args
        ):
            args[key] = value
    return args


class DCELoss(torch.nn.Module):
    """Disagreement Cross Entropy Loss."""

    def __init__(self, weight=None, use_random_vectors=False, alpha=None):
        super(DCELoss, self).__init__()
        self.weight = weight
        self.use_random_vectors = use_random_vectors
        self.alpha = alpha

    def forward(self, logits, labels, mask):
        """Forward pass of the loss function."""
        return dce_loss(
            logits,
            labels,
            mask,
            alpha=self.alpha,
            use_random_vectors=self.use_random_vectors,
            weight=self.weight,
        )


def dce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha: Optional[float] = None,
    use_random_vectors=False,
    weight=None,
) -> torch.Tensor:
    """
    Disagreement Cross Entropy Loss functional.

    :param logits: (batch_size, num_classes) tensor of logits
    :param labels: (batch_size,) tensor of labels
    :param mask: (batch_size,) mask
    :param alpha: (float) weight of q samples
    :param use_random_vectors: (bool) whether to use
           random vectors for negative labels, default=False
    :param weight:  (torch.Tensor) weight for each sample_data,
                    default=None do not apply weighting
    :return: (tensor, float) the disagreement cross entropy loss
    """
    if mask.all():
        # if all labels are positive, then use the standard cross entropy loss
        # infer multi-label classification from the dtype of labels
        if labels.dtype == torch.float32:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss
    if alpha is None:
        alpha = 1 / (1 + (~mask).float().sum())

    num_classes = logits.shape[1]

    q_logits, q_labels = logits[~mask], labels[~mask]
    if use_random_vectors:
        # noinspection PyTypeChecker,PyUnresolvedReferences
        p = -torch.log(
            torch.rand(device=q_labels.device, size=(len(q_labels), num_classes)),
        )
        p *= 1.0 - torch.nn.functional.one_hot(q_labels, num_classes=num_classes)
        p /= torch.sum(p)
        ce_n = -(p * q_logits).sum(1) + torch.logsumexp(q_logits, dim=1)

    else:
        if labels.dtype == torch.long:
            zero_hot = 1.0 - torch.nn.functional.one_hot(
                q_labels,
                num_classes=num_classes,
            )
        else:
            zero_hot = 1.0 - q_labels
        ce_n = -(q_logits * zero_hot).sum(dim=1) / (num_classes - 1) + torch.logsumexp(
            q_logits,
            dim=1,
        )

    if torch.isinf(ce_n).any() or torch.isnan(ce_n).any():
        raise RuntimeError("NaN or Infinite loss encountered for ce-q")

    if (~mask).all():
        return (ce_n * alpha).mean()

    p_logits, p_labels = logits[mask], labels[mask]
    if labels.dtype == torch.float32:
        ce_p = torch.nn.functional.binary_cross_entropy_with_logits(
            p_logits,
            p_labels,
            reduction="none",
            weight=weight,
        )
    else:
        ce_p = torch.nn.functional.cross_entropy(
            p_logits,
            p_labels,
            reduction="none",
            weight=weight,
        )
    return torch.cat([ce_n * alpha, ce_p]).mean()


class DetectronModule(nn.Module):
    """Detectron wrapper module."""

    def __init__(self, model: nn.Module, feature_column: str, alpha=None):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.feature_column = feature_column
        self.criterion = DCELoss(alpha=self.alpha)

    def forward(self, **kwargs):
        """Forward pass of the model."""
        labels = kwargs.pop("labels", None)
        mask = kwargs.pop("mask", None)
        x = kwargs.pop(self.feature_column)
        logits = self.model(x)
        return logits if labels is None else self.criterion(logits, labels, mask)


class DummyCriterion(nn.Module):
    """Dummy criterion."""

    def __init__(self):
        super().__init__()

    def forward(self, loss, labels):
        """Forward pass of the criterion."""
        return loss
