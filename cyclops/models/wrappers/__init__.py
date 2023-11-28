"""Wrapper classes for models."""

from typing import Union

from cyclops.models.wrappers.sk_model import SKModel
from cyclops.utils.optional import import_optional_module


torch = import_optional_module("torch", error="warn")
if torch is not None:
    from cyclops.models.wrappers.pt_model import PTModel
else:
    PTModel = None

WrappedModel = Union[PTModel, SKModel]
