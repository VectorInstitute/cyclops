"""Tasks utility functions."""
from typing import Dict, List, Mapping, Sequence, Union, get_args

import numpy as np
import pandas as pd
import PIL
import torch
import yaml
from torchvision.transforms import PILToTensor

from cyclops.models.catalog import create_model
from cyclops.models.constants import CONFIG_ROOT
from cyclops.models.wrappers import WrappedModel
from cyclops.utils.file import join

CXR_TARGET = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]


def apply_image_transforms(examples: Dict[str, List], transforms: callable) -> dict:
    """Apply transforms to examples.

    Used for applying image transformations to examples for chest X-ray classification.

    """
    # examples is a dict of lists; convert to list of dicts.
    # doing a conversion from PIL to tensor is necessary here when working
    # with the Image feature type.
    value_len = len(list(examples.values())[0])
    examples = [
        {
            k: PILToTensor()(v[i]) if isinstance(v[i], PIL.Image.Image) else v[i]
            for k, v in examples.items()
        }
        for i in range(value_len)
    ]

    # apply the transforms to each example
    examples = [transforms(example) for example in examples]

    # convert back to a dict of lists
    examples = {k: [d[k] for d in examples] for k in examples[0]}

    return examples


def to_numpy(X) -> np.ndarray:
    """Convert input to a numpy array.

    Parameters
    ----------
    X : Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series, Mapping, Sequence]
        Input data.

    Returns
    -------
    np.ndarray
        Output numpy array.

    Raises
    ------
    ValueError
        Input type is not supported.

    """
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()

    if isinstance(X, torch.Tensor):
        if X.requires_grad:
            X = X.detach()

        if X.is_cuda:
            X = X.cpu()
        return X.numpy()

    if np.isscalar(X):
        return np.array(X)

    if isinstance(X, Sequence):
        return type(X)(to_numpy(x) for x in X)
    if isinstance(X, Mapping):
        return {k: to_numpy(v) for k, v in X.items()}

    raise ValueError(
        "Cannot convert to numpy array. `X` must be a numpy array, torch tensor,"
        f" dictionary, list, tuple, pandas dataframe or pandas series. \
        Got {type(X)} instead."
    )


def prepare_models(
    models: Union[
        str, WrappedModel, Sequence[Union[str, WrappedModel]], Dict[str, WrappedModel]
    ],
    config_path: Union[str, Dict[str, str]] = None,
) -> Dict[str, WrappedModel]:
    """Prepare the models as a dictionary, and wrap those that are not wrapped.

    Parameters
    ----------
    models : Union[
        str,
        WrappedModel,
        Sequence[Union[str, WrappedModel]],
        Dict[str, WrappedModel]
    ]
        model name(s) or wrapped model(s) or a dictionary
        of model name(s) and wrapped model(s).
    config_path : Union[str, Dict[str, str]], optional
        Path to the configuration file(s) for the model(s),
        by default None

    Returns
    -------
    Dict[str, WrappedModel]
        Dictionary model names and wrapped models

    Raises
    ------
    TypeError
        Models in a list are not of type WrappedModel or string.
    TypeError
        Models are not of the types indicated above.

    """
    models_dict = {}

    if isinstance(models, get_args(WrappedModel)):
        model_name = models.model.__name__
        models_dict = {model_name: models}
    elif isinstance(models, str):
        config_path = (
            config_path if config_path else join(CONFIG_ROOT, models + ".yaml")
        )
        with open(config_path, "r", encoding="utf8") as file:
            config = yaml.safe_load(file)
            config = config.get("model_params", None) or config
        models_dict = {models: create_model(models, **config)}
    elif isinstance(models, (list, tuple)):
        for model in models:
            if isinstance(model, get_args(WrappedModel)):
                models_dict[model.model.__name__] = model
            elif isinstance(model, str):
                config_path = (
                    config_path[model]
                    if config_path and model in config_path
                    else join(CONFIG_ROOT, model + ".yaml")
                )
                with open(config_path, "r", encoding="utf8") as file:
                    config = yaml.safe_load(file)
                    config = config.get("model_params", None) or config
                    print(config)
                models_dict[model] = create_model(model, **config)
            else:
                raise TypeError(
                    "models must be lists or tuples of strings or \
                    PTModel instances or SKModel instances."
                )
    elif isinstance(models, dict):
        assert all(isinstance(m, get_args(WrappedModel)) for m in models.values())
        models_dict = models
    else:
        raise TypeError(f"Invalid model type: {type(models)}")

    return models_dict
