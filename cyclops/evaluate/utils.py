"""Utility functions for the `evaluate` package."""

from typing import Any, List, Mapping, Union

from datasets import DatasetDict, IterableDatasetDict, get_dataset_split_names


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
