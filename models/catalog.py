"""Model catalog."""

from typing import Optional, Callable

from models.mlp import MLP


MODEL_CATALOG = {}


def register_model(
    register_dict: dict, model: type, name: Optional[str] = None
) -> Callable:
    """Register model with dict.

    Parameters
    ----------
    register_dict: dict
        Dictionary to store registered model implementations.
    model: type
        Model implementation wrapped in a class.
    name: str, optional
    """
    if not name:
        name = model.__name__
    if name not in MODEL_CATALOG:
        MODEL_CATALOG[name] = model


def get_model(name: str) -> type:
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
    else:
        return MODEL_CATALOG.get(name)


# Register model implmentations.
register_model(MODEL_CATALOG, MLP)
