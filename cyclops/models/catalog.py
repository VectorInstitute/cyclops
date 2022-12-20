"""Model catalog."""

import logging
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Literal, Set, Union

import torch.nn.modules
import torch.optim
from sklearn.base import BaseEstimator

from cyclops.models.utils import is_pytorch_model, is_sklearn_model
from cyclops.models.wrappers import PTModel, SKModel, WrappedModel
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)

####################
# Model catalogs   #
####################
_model_catalog: Dict[str, Any] = {}
_temporal_model_keys: Set[str] = set()
_static_model_keys: Set[str] = set()
_pt_model_keys: Set[str] = set()
_sk_model_keys: Set[str] = set()


def register_model(name: str, model_type: Literal["static", "temporal"]) -> Callable:
    """Register model in the catalog.

    Parameters
    ----------
    name : str
        The name of the model.
    model_type : "static", "temporal"
        The temporal or static nature of the model.

    Returns
    -------
    Callable
        A decorator that registers the model.

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
) -> List[str]:
    """List models.

    Parameters
    ----------
    category : "static", "temporal", "pytorch", "sklearn", optional
        The type of model to list. If None, all models are listed.

    Returns
    -------
    model_list : list[str]
        List of model names.

    """
    if category is None:
        model_list = list(_model_catalog.keys())
    elif category == "static":
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


def wrap_model(model: Union[torch.nn.Module, BaseEstimator], **kwargs) -> WrappedModel:
    """Wrap a model with SKModel or PTModel.

    Parameters
    ----------
    model : Union[torch.nn.Module, sklearn.base.BaseEstimator]
        The model to wrap.

    Returns
    -------
    SKModel or PTModel
        The wrapped model.

    Raises
    ------
    TypeError
        If model is not a pyTorch or sklearn model.

    """
    if is_pytorch_model(model):
        return PTModel(model, **kwargs)
    if is_sklearn_model(model):
        return SKModel(model, **kwargs)
    raise TypeError("``model`` must be a pyTorch or sklearn model")


def create_model(model_name: str, wrap: bool = True, **kwargs) -> WrappedModel:
    """Create model and optionally wrap it.

    Parameters
    ----------
    model_name : str
        Model name.
    wrap : bool, optional
        Whether to wrap model.
    **kwargs : dict, optional
        Keyword arguments passed to the wrapper class or model class.

    Returns
    -------
    model : PTModel or SKModel
        An instance of the model.

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
