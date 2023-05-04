"""Utilities for datasets."""
from typing import Any, Callable, Dict, List, Optional, Union, get_args

import numpy as np
import numpy.typing as npt
import PIL
import psutil
from datasets import Dataset
from datasets.features import (
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    ClassLabel,
    Sequence,
    Value,
)
from numpy.typing import ArrayLike
from torchvision.transforms import PILToTensor

NUMERIC_FEATURE_TYPES = (
    "bool",
    "int",
    "uint",
    "float",
    "decimal",
)
DATETIME_FEATURE_TYPES = (
    "date",
    "duration",
    "time",
    "timestamp",
)
FEATURE_TYPES = Union[  # pylint: disable=invalid-name
    ClassLabel,
    Sequence,
    Value,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
]


def set_decode(
    dataset: Dataset, decode: bool = True, exclude: Optional[List[str]] = None
) -> None:
    """Set decode attribute of dataset features that have it.

    Parameters
    ----------
    dataset : Dataset
       A Hugging Face dataset object.
    decode : bool, optional, default=True
        Whether to set decode attribute to True or False for features that have
        it.
    exclude : List[str], optional, default=None
        List of feature names to exclude. If None, no features are excluded.
        An example of when this might be useful is when dealing with image
        segmentation tasks where the target and prediction are images that need
        to be decoded, whereas the original image may not need to be decoded.

    """
    assert isinstance(dataset, Dataset), "dataset must be a Hugging Face dataset"
    if exclude is not None:
        if not isinstance(exclude, list) or not all(
            feature in dataset.column_names for feature in exclude
        ):
            raise ValueError(
                "`exclude` must be a list of feature names that are present in "
                f"dataset. Got {exclude} of type `{type(exclude)}` and dataset "
                f"with columns {dataset.column_names}."
            )

    for feature_name, feature in dataset.features.items():
        if feature_name not in (exclude or []) and hasattr(feature, "decode"):
            dataset.features[feature_name].decode = decode


def get_columns_as_numpy_array(
    dataset: Union[Dataset, Dict[str, ArrayLike]], columns: Union[str, List[str]]
) -> npt.NDArray[Any]:
    """Get columns of dataset as numpy array.

    Parameters
    ----------
    dataset : Dataset, Dict[str, ArrayLike]
        A Hugging Face dataset object or a dictionary of arraylike objects.
    columns : List[str], str
        List of column names or single column name to get as numpy array.

    Returns
    -------
    np.ndarray
        Numpy array of columns.

    """
    if not isinstance(dataset, (Dataset, dict)):
        raise TypeError(
            "dataset must be a Hugging Face dataset or a dictionary of numpy arrays."
        )

    if isinstance(columns, str):
        columns = [columns]

    if isinstance(dataset, Dataset) and dataset.format != "numpy":
        with dataset.formatted_as("numpy", columns=columns, output_all_columns=True):
            return np.stack(  # type: ignore[no-any-return]
                [dataset[col] for col in columns], axis=-1
            ).squeeze()

    return np.stack([dataset[col] for col in columns], axis=-1).squeeze()


def check_required_columns(
    dataset_column_names: List[str], *required_columns: Union[List[str], str, None]
) -> None:
    """Check if required columns are present in dataset.

    Parameters
    ----------
    dataset_column_names : List[str]
        List of column names in dataset.
    required_columns : Union[List[str], str, None]
        List of required column names or single required column name.

    Raises
    ------
    ValueError
        If a required column is not present in the dataset.

    """
    required_columns_ = []
    for required_column in required_columns:
        if required_column is None:
            continue
        if isinstance(required_column, str):
            required_columns_.append(required_column)
        else:
            required_columns_.extend(required_column)

    for column in required_columns_:
        if column is not None and column not in dataset_column_names:
            raise ValueError(
                f"Column {column} is not present in the dataset. Please "
                "specify a valid column. The following columns are present "
                f"in the dataset: {dataset_column_names}."
            )


def feature_is_numeric(feature: FEATURE_TYPES) -> bool:
    """Check if Hugging Face dataset feature is numeric.

    Parameters
    ----------
    feature : Union[ClassLabel, Sequence, Value, Array2D, Array3D, Array4D, Array5D]
        Hugging Face dataset feature.

    Returns
    -------
    bool
        True if feature is numeric, False otherwise.

    Raises
    ------
    TypeError
        If `feature` is not a valid Hugging Face dataset feature.

    """
    dtype = feature.dtype
    if dtype == "list":  # Sequence
        return all(feature_is_numeric(subfeature) for subfeature in feature.feature)
    if not isinstance(feature, get_args(FEATURE_TYPES)):
        raise TypeError(f"Invalid type for `feature`: {type(feature)}.")

    return any(dtype.startswith(t) for t in NUMERIC_FEATURE_TYPES)


def feature_is_datetime(feature: FEATURE_TYPES) -> bool:
    """Check if Hugging Face dataset feature is datetime.

    Parameters
    ----------
    feature : Union[ClassLabel, Sequence, Value, Array2D, Array3D, Array4D, Array5D]
        Hugging Face dataset feature.

    Returns
    -------
    bool
        True if feature is datetime, False otherwise.

    Raises
    ------
    TypeError
        If `feature` is not a valid Hugging Face dataset feature.

    """
    dtype = feature.dtype
    if dtype == "list":  # Sequence
        return all(feature_is_datetime(subfeature) for subfeature in feature.feature)
    if not isinstance(feature, get_args(FEATURE_TYPES)):
        raise TypeError(f"Invalid type for `feature`: {type(feature)}.")

    return any(dtype.startswith(t) for t in DATETIME_FEATURE_TYPES)


def is_out_of_core(dataset_size: int) -> Any:
    """Check if dataset is too large to fit in memory.

    Parameters
    ----------
    dataset_size : int
        Size of dataset expressed in bytes

    Returns
    -------
    Any
        Whether dataset can fit in memory

    """
    return dataset_size > psutil.virtual_memory().available


def apply_transforms(
    examples: Dict[str, Any], transforms: Callable[..., Any]
) -> Dict[str, Any]:
    """Apply transforms to examples."""
    # examples is a dict of lists; convert to list of dicts.
    # doing a conversion from PIL to tensor is necessary here when working
    # with the Image feature type.
    value_len = len(list(examples.values())[0])
    examples_list = [
        {
            k: PILToTensor()(v[i]) if isinstance(v[i], PIL.Image.Image) else v[i]
            for k, v in examples.items()
        }
        for i in range(value_len)
    ]

    # apply the transforms to each example
    examples_list = [transforms(example) for example in examples_list]

    # convert back to a dict of lists
    examples = {k: [d[k] for d in examples_list] for k in examples_list[0]}

    return examples
