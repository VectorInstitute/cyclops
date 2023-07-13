"""Utility functions for the `cyclops.report` module."""
import importlib
import inspect
from re import sub
from typing import Any, Mapping

from cyclops.report.model_card import ModelCard  # type: ignore[attr-defined]


def str_to_snake_case(string: str) -> str:
    """Convert a string to snake_case.

    Parameters
    ----------
    string : str
        The string to convert.

    Returns
    -------
    str
        The converted string.

    Examples
    --------
    >>> str_to_snake_case("HelloWorld")
    'hello_world'
    >>> str_to_snake_case("Hello-World")
    'hello_world'
    >>> str_to_snake_case("Hello_World")
    'hello_world'
    >>> str_to_snake_case("Hello World")
    'hello_world'
    >>> str_to_snake_case("hello_world")
    'hello_world'

    """
    string = "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", string.replace("-", " "))
        ).split()
    ).lower()

    return string


def _raise_if_not_dict_with_str_keys(data: Any) -> None:
    """Raise an error if `data` is not a dictionary with string keys.

    Parameters
    ----------
    data : Any
        The data to check.

    Raises
    ------
    TypeError
        If `data` is not a dictionary with string keys.

    """
    if not (
        isinstance(data, Mapping) and all(isinstance(key, str) for key in data.keys())
    ):
        raise TypeError(f"Expected a dictionary with string keys. Got {data} instead.")


def _object_is_in_model_card_module(obj: object) -> bool:
    """Check if an object is defined in the same module as `ModelCard`.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        Whether or not the object is defined in the same module as `ModelCard`.

    """
    model_card_module = importlib.import_module(ModelCard.__module__)
    model_card_classes = inspect.getmembers(model_card_module, inspect.isclass)
    for name, model_card_class in model_card_classes:
        # match name or class
        if model_card_class.__module__ == ModelCard.__module__ and (
            obj.__class__.__name__ == name or obj.__class__ == model_card_class
        ):
            return True
    return False
