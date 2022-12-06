"""Model catalog."""
import logging
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Literal, TypeVar, Union

from cyclops.utils.log import setup_logging
from models.utils import is_pytorch_model, is_sklearn_model
from models.wrapper import PTModel, SKModel, wrap_model

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)

_Model = TypeVar("_Model", PTModel, SKModel)

_model_catalog: Dict[str, Any] = {}
_temporal_model_keys: set[str] = set()
_static_model_keys: set[str] = set()
_pt_model_keys: set[str] = set()
_sk_model_keys: set[str] = set()


def register_model(name: str, model_type: Literal["static", "temporal"]) -> Callable:
    """Register model in the catalog.

    Parameters
    ----------
    name: str
        Model name.
    model_type: Literal["static", "temporal"]
        Model type.

    Returns
    -------
    Callable
        Decorator.

    Raises
    ------
    NotImplementedError
        If model type or model library is not supported.

    """

    def decorator(model_obj: type) -> type:
        if name in _model_catalog:
            LOGGER.warning(
                "Model with name %s is already registered. "
                "It will be replaced by the new model.",
                name,
            )

        _model_catalog[name] = model_obj

        if model_type == "static":
            _static_model_keys.add(name)
        elif model_type == "temporal":
            _temporal_model_keys.add(name)
        else:
            raise NotImplementedError(f"Model type {model_type} is not supported.")

        # infer model library
        if is_pytorch_model(model_obj):
            _pt_model_keys.add(name)
        elif is_sklearn_model(model_obj):
            _sk_model_keys.add(name)
        else:
            raise NotImplementedError(
                "Model library is not supported. Only PyTorch and scikit-learn "
                " mmodels are supported."
            )

        return model_obj

    return decorator


def list_models(
    category: Literal["static", "temporal", "pytorch", "sklearn"] = None
) -> list[str]:
    """List models.

    Parameters
    ----------
    category: Literal["static", "temporal", "pytorch", "sklearn"], optional
        Model category.

    Returns
    -------
    list[str]
        List of models.

    """
    if category is None:
        model_list = list(_model_catalog.keys())
    if category == "static":
        model_list = list(_static_model_keys)
    elif category == "temporal":
        model_list = list(_temporal_model_keys)
    elif category == "pytorch":
        model_list = list(_pt_model_keys)
    elif category == "sklearn":
        model_list = list(_sk_model_keys)
    else:
        raise ValueError(
            f"Category {category} not supported."
            " Choose from: `static`, `temporal`, `pytorch` or `sklearn`."
        )

    return model_list


def create_model(
    model_name: str, wrap: bool = True, **kwargs
) -> Union[SKModel, PTModel]:
    """Create model.

    Parameters
    ----------
    model_name: str
        Model name.
    wrap: bool, optional
        Whether to wrap model.
    kwargs
        Keyword arguments passed to the wrapper class or model class.

    Returns
    -------
    Union[_Model, type]
        Model.

    """
    model_class = _model_catalog.get(model_name, None)
    if model_class is None:
        similar_keys_list: List[str] = get_close_matches(
            model_name, _model_catalog.keys(), n=5
        )
        similar_keys: str = ", ".join(similar_keys_list)
        similar_keys = (
            f" Did you mean one of: {similar_keys}?"
            if similar_keys
            else "It may not be in the catalog."
        )
        raise ValueError(f"Model {model_name} not found.{similar_keys}")

    if wrap:
        model = wrap_model(model_class, **kwargs)
    else:
        model = model_class(**kwargs)

    return model
