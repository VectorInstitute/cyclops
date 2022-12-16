"""Wrapper classes for models."""
from typing import Union

from models.wrappers.pt_model import PTModel
from models.wrappers.sk_model import SKModel

WrappedModel = Union[PTModel, SKModel]
