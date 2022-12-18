"""Wrapper classes for models."""
from typing import Union

from cyclops.models.wrappers.pt_model import PTModel
from cyclops.models.wrappers.sk_model import SKModel

WrappedModel = Union[PTModel, SKModel]
