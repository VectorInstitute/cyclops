"""Tasks utility functions."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence, Union, get_args

import PIL

from cyclops.models.catalog import _model_names_mapping, create_model, list_models
from cyclops.models.wrappers import WrappedModel
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    from torchvision.transforms import PILToTensor
else:
    PILToTensor = import_optional_module(
        "torchvision.transforms",
        attribute="PILToTensor",
        error="warn",
    )
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


def apply_image_transforms(
    examples: Dict[str, List[Any]],
    transforms: Callable[[Any], Any],
) -> Dict[str, List[Any]]:
    """Apply transforms to examples.

    Used for applying image transformations to examples for chest X-ray classification.

    """
    # examples is a dict of lists; convert to list of dicts.
    # doing a conversion from PIL to tensor is necessary here when working
    # with the Image feature type.
    value_len = len(list(examples.values())[0])
    examples_ = [
        {
            k: PILToTensor()(v[i]) if isinstance(v[i], PIL.Image.Image) else v[i]
            for k, v in examples.items()
        }
        for i in range(value_len)
    ]

    # apply the transforms to each example
    examples_ = [transforms(example) for example in examples_]

    # convert back to a dict of lists
    return {k: [d[k] for d in examples_] for k in examples_[0]}


def prepare_models(
    models: Union[
        str,
        WrappedModel,
        Sequence[Union[str, WrappedModel]],
        Dict[str, WrappedModel],
    ],
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
    # models contains one wrapped model (SKModel or PTModel)
    if isinstance(models, get_args(WrappedModel)):
        model_name = _model_names_mapping.get(models.model.__name__)
        models_dict = {model_name: models}
    # models contains one model name
    elif isinstance(models, str):
        assert models in list_models(), f"Model name is not registered! \
                    Available models are: {list_models()}"
        models_dict = {models: create_model(models)}
    # models contains a list or tuple of model names or wrapped models
    elif isinstance(models, (list, tuple)):
        for model in models:
            if isinstance(model, get_args(WrappedModel)):
                model_name = _model_names_mapping.get(model.model.__name__)
                models_dict[model_name] = model
            elif isinstance(model, str):
                assert model in list_models(), f"Model name is not registered! \
                    Available models are: {list_models()}"
                models_dict[model] = create_model(model)
            else:
                raise TypeError(
                    "models must be lists/tuples of strings,\
                    PTModel instances or SKModel instances.",
                )
    # models contains a dictionary of model names and wrapped models
    elif isinstance(models, dict):
        assert all(isinstance(m, get_args(WrappedModel)) for m in models.values())
        models_dict = models  # type: ignore
    else:
        raise TypeError(f"Invalid model type: {type(models)}")

    return models_dict  # type: ignore
