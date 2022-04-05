"""Model catalog."""

from typing import Optional

from models.mlp import MLP

MODEL_CATALOG = {}


def register_model(model: type, name: Optional[str] = None):
    """Register model with dict.

    Parameters
    ----------
    model: type
        Model implementation wrapped in a class.
    name: str, optional
    """
    if not name:
        name = model.__name__
    if name not in MODEL_CATALOG:
        MODEL_CATALOG[name] = model


def get_model(name: str) -> Optional[type]:
    """Get model from catalog.

    Parameters
    ----------
    name: str
        Model name.

    Returns
    -------
    type
        Model class.

    Raises
    ------
    NotImplementedError
        If model name provided is not in the catalog.
    """
    if name not in MODEL_CATALOG:
        raise NotImplementedError
    return MODEL_CATALOG.get(name)


# Register model implementations.
register_model(MLP)
