"""Model catalog."""

import logging
import os
from difflib import get_close_matches
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Union,
)

import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.base import BaseEstimator

from cyclops.models.utils import is_pytorch_model, is_sklearn_model
from cyclops.models.wrappers import SKModel, WrappedModel
from cyclops.utils.file import join
from cyclops.utils.log import setup_logging
from cyclops.utils.optional import import_optional_module


CONFIG_ROOT = join(os.path.dirname(__file__), "configs")
if TYPE_CHECKING:
    import torch
    from torch.nn import Module
else:
    torch = import_optional_module("torch", error="warn")
    Module = import_optional_module("torch.nn", attribute="Module", error="warn")


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)
_xgboost_unavailable_message = (
    "The XGBoost library is required to use the `XGBClassifier` model. "
    "Please install it as an extra using `python3 -m pip install 'pycyclops[xgboost]'`\
        or using `python3 -m pip install xgboost`."
)
_torchxrayvision_unavailable_message = (
    "The torchxrayvision library is required to use the `densenet` or `resnet` model. "
    "Please install it as an extra using `python3 -m pip install 'pycyclops[torchxrayvision]'`\
        or using `python3 -m pip install torchxrayvision`."
)
_torch_unavailable_message = (
    "The PyTorch library is required to use the `mlp_pt`, `gru`, `lstm` or `rnn` models. "
    "Please install it as an extra using `python3 -m pip install 'pycyclops[torch]'`\
        or using `python3 -m pip install torch`."
)

####################
# Model catalogs   #
####################
_model_catalog: Dict[str, Any] = {}
_model_names_mapping: Dict[str, str] = {}
_temporal_model_keys: Set[str] = set()
_static_model_keys: Set[str] = set()
_img_model_keys: Set[str] = set()
_pt_model_keys: Set[str] = set()
_sk_model_keys: Set[str] = set()


def register_model(
    name: str,
    model_type: Literal["static", "temporal", "image"],
) -> Callable:
    """Register model in the catalog.

    Parameters
    ----------
    name : str
        The name of the model.
    model_type : "static", "temporal", "image"
        The type of model.

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
        _model_names_mapping[model_obj.__name__] = name

        if model_type == "static":
            _static_model_keys.add(name)
        elif model_type == "temporal":
            _temporal_model_keys.add(name)
        elif model_type == "image":
            _img_model_keys.add(name)
        else:
            raise NotImplementedError(f"Model type {model_type} is not supported.")

        # infer model library
        if is_sklearn_model(model_obj):
            _sk_model_keys.add(name)
        elif is_pytorch_model(model_obj):
            _pt_model_keys.add(name)
        else:
            raise NotImplementedError(
                "Model library is not supported. Only PyTorch and scikit-learn "
                " mmodels are supported.",
            )

        return model_obj

    return decorator


def list_models(
    category: Optional[
        Literal["static", "temporal", "image", "pytorch", "sklearn"]
    ] = None,
) -> List[str]:
    """List models.

    Parameters
    ----------
    category : "static", "temporal", "image", "pytorch", "sklearn", optional
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
    elif category == "image":
        model_list = list(_img_model_keys)
    elif category == "pytorch":
        model_list = list(_pt_model_keys)
    elif category == "sklearn":
        model_list = list(_sk_model_keys)
    else:
        raise ValueError(
            f"Category {category} not supported."
            " Choose from: `static`, `temporal`, `pytorch` or `sklearn`.",
        )

    return model_list


def wrap_model(model: Union[Module, BaseEstimator], **kwargs) -> WrappedModel:
    """Wrap a model with SKModel or PTModel.

    Parameters
    ----------
    model : Union[torch.nn.Module, sklearn.base.BaseEstimator]
        The model to wrap.
    **kwargs : dict, optional
        Keyword arguments passed to the wrapper class.

    Returns
    -------
    SKModel or PTModel
        The wrapped model.

    Raises
    ------
    TypeError
        If model is not a pyTorch or sklearn model.

    """
    if is_sklearn_model(model):
        return SKModel(model, **kwargs)
    if is_pytorch_model(model):
        from cyclops.models.wrappers import PTModel

        return PTModel(model, **kwargs)
    raise TypeError("``model`` must be a PyTorch or sklearn model")


def create_model(
    model_name: str,
    wrap: bool = True,
    **config_overrides,
) -> WrappedModel:
    """Create model and optionally wrap it.

    Parameters
    ----------
    model_name : str
        Model name.
    wrap : bool, optional
        Whether to wrap model.
    **config_overrides : dict, optional
        Keyword arguments passed to the wrapper class or model class \
            to override the predefined config.

    Returns
    -------
    model : PTModel or SKModel
        An instance of the model.

    """
    model_class = _model_catalog.get(model_name)
    if model_class is None:
        if model_name == "xgb_classifier":
            raise RuntimeError(_xgboost_unavailable_message)
        if model_name in ["densenet", "resnet"]:
            raise RuntimeError(_torchxrayvision_unavailable_message)
        if model_name in ["gru", "lstm", "mlp_pt", "rnn"]:
            raise RuntimeError(_torch_unavailable_message)
        similar_keys_list: List[str] = get_close_matches(
            model_name,
            _model_catalog.keys(),
            n=5,
        )
        similar_keys: str = ", ".join(similar_keys_list)
        similar_keys = (
            f" Did you mean one of: {similar_keys}?"
            if similar_keys
            else "It may not be in the catalog."
        )
        raise ValueError(f"Model {model_name} not found. {similar_keys}")

    overrides = []
    if config_overrides:
        config_file = join(CONFIG_ROOT, f"{model_name}.yaml")
        with open(config_file, "r", encoding="utf-8") as file:
            config_keys = list(yaml.safe_load(file).keys())
        for key, value in config_overrides.items():
            if key in config_keys:
                overrides.append(f"{key}={value}")
            else:
                overrides.append(f"+{key}={value}")
    with initialize(version_base=None, config_path="configs", job_name="create_model"):
        config = compose(config_name=f"{model_name}.yaml", overrides=overrides)
        LOGGER.debug(OmegaConf.to_yaml(config))

    return wrap_model(model_class, **config) if wrap else model_class(**config)
