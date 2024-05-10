"""Utility functions for the `evaluate` package."""

import logging
from contextlib import nullcontext
from typing import Any, List, Literal, Mapping, Union

import array_api_compat.numpy
import pyarrow as pa
from datasets import Dataset, DatasetDict, IterableDatasetDict, get_dataset_split_names

from cyclops.evaluate.metrics.experimental.utils.ops import squeeze_all
from cyclops.evaluate.metrics.experimental.utils.types import Array
from cyclops.utils.log import setup_logging
from cyclops.utils.optional import import_optional_module


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


_SUPPORTED_ARRAY_LIBS = {"numpy": array_api_compat.numpy}

cupy = import_optional_module("array_api_compat.cupy", error="warn")
torch = import_optional_module("array_api_compat.torch", error="warn")
if cupy is not None:
    _SUPPORTED_ARRAY_LIBS["cupy"] = cupy
if torch is not None:
    _SUPPORTED_ARRAY_LIBS["torch"] = torch


def check_required_columns(
    dataset_column_names: List[str],
    *required_columns: Union[List[str], str, None],
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
    required_columns_ = [
        column
        for column in required_columns
        if column is not None
        for column in (column if isinstance(column, list) else [column])
        if column is not None
    ]
    missing_columns = set(required_columns_) - set(dataset_column_names)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing the following required columns: {missing_columns}.",
        )


def choose_split(
    dataset: Union[str, DatasetDict, IterableDatasetDict],
    **kwargs: Mapping[str, Any],
) -> str:
    """Choose a dataset split to use for evaluation if none is provided.

    Parameters
    ----------
    dataset : Union[str, DatasetDict, IterableDatasetDict]
        Dataset to choose a split from.
    **kwargs : Mapping[str, Any]
        Keyword arguments to pass to `get_dataset_split_names` if `dataset`
        is a string.

    Returns
    -------
    str
        Name of the chosen split.

    Raises
    ------
    ValueError
        If `split` is `None` and no split can be chosen.

    """
    if isinstance(dataset, str):
        # change `name` to `config_name` in kwargs
        if kwargs is None:
            kwargs = {}
        if "name" in kwargs:
            kwargs["config_name"] = kwargs.pop("name")
        available_splits: List[str] = get_dataset_split_names(dataset, **kwargs)
    elif isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        available_splits = list(dataset.keys())

    preferred_split_order = [
        "test",
        "testing",
        "eval",
        "evaluation",
        "validation",
        "val",
        "valid",
        "dev",
        "train",
        "training",
    ]

    for split in preferred_split_order:
        if split in available_splits:
            return split

    raise ValueError(
        "No dataset split defined! Pass an explicit value to the `split` kwarg.",
    )


def get_columns_as_array(
    dataset: Union[Dataset, pa.Table],
    columns: Union[str, List[str]],
    array_lib: Literal["torch", "numpy", "cupy"] = "numpy",
) -> Array:
    """Get columns of dataset as array.

    Parameters
    ----------
    dataset : Dataset, pyarrow.Table
        A HuggingFace dataset or a pyarrow Table object representing the dataset
        to get the columns from.
    columns : List[str], str
        List of column names or single column name to get as an array.
    array_lib : {"torch", "numpy", "cupy"}, default="numpy"
        The array library to get the column(s) as.

    Returns
    -------
    Array
        The array of columns.

    Raises
    ------
    TypeError
        If `dataset` is not a Huggingface dataset or a pyarrow.Table object.

    """
    if not isinstance(dataset, (Dataset, pa.Table)):
        raise TypeError(
            "`dataset` must be a HuggingFace dataset or a pyarrow.Table object.",
        )
    if array_lib not in _SUPPORTED_ARRAY_LIBS:
        raise NotImplementedError(f"The array library `{array_lib}` is not supported.")

    xp = _SUPPORTED_ARRAY_LIBS[array_lib]

    if isinstance(columns, str):
        columns = [columns]

    with dataset.formatted_as("arrow", columns=columns, output_all_columns=True) if (
        isinstance(dataset, Dataset) and dataset.format != "arrow"
    ) else nullcontext():
        out_arr = squeeze_all(
            xp.stack(
                [xp.asarray(dataset[col].to_pylist()) for col in columns], axis=-1
            ),
        )

    if out_arr.ndim == 0:
        out_arr = xp.expand_dims(out_arr, axis=-1)

    return out_arr


def _format_column_names(column_names: Union[str, List[str]]) -> List[str]:
    """Format the column names to list of strings if not already a list.

    Parameters
    ----------
    column_names : Union[str, List[str]]
        The column names to format.

    Returns
    -------
    List[str]
        The formatted column names.

    Raises
    ------
    TypeError
        If any of the column names are not strings or list of strings.

    """
    if isinstance(column_names, str):
        return [column_names]
    if isinstance(column_names, list):
        return column_names

    raise TypeError(
        f"Expected column name {column_names} to be a string or "
        f"list of strings, but got {type(column_names)}.",
    )
