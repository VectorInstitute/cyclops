"""Utilities for handling optional dependencies."""

import importlib
import importlib.util
import warnings
from types import ModuleType
from typing import Literal, Optional, Union


def import_optional_module(
    name: str,
    attribute: Optional[str] = None,
    error: Literal["raise", "warn", "ignore"] = "raise",
) -> Union[ModuleType, None]:
    """Import an optional module.

    Parameters
    ----------
    name : str
        The name of the module to import.
    attribute : Optional[str], optional
        The name of an attribute to import from the module.
    error : ErrorOption, optional
        How to handle errors. One of:
        - "raise": raise an error if the module cannot be imported.
        - "warn": raise a warning if the module cannot be imported.
        - "ignore": ignore the missing module and return `None`.

    Returns
    -------
    ModuleType or None
        None if the module could not be imported,
        or the module or attribute if it was imported successfully.

    Raises
    ------
    ImportError
        If the module could not be imported and `error` is set to "raise".

    Warns
    -----
    UserWarning
        If the module could not be imported and `error` is set to "warn".

    Notes
    -----
    This function is useful for handling optional dependencies. It will
    attempt to import the specified module and return it if it is found.
    If the module is not found, it will raise an ImportError, raise a
    warning, or return ``None`` based on the value of
    the `error` parameter.

    """
    if error not in ("raise", "warn", "ignore"):
        raise ValueError(
            "Expected `error` to be one of 'raise', 'warn, or 'ignore', "
            f"but got {error}.",
        )

    try:
        module = importlib.import_module(name)
        if attribute is not None and module is not None:
            module = getattr(module, attribute)
        return module
    except ModuleNotFoundError as exc:
        msg = (
            f"Missing optional dependency '{name}'. "
            f"Use pip or conda to install {name}."
        )
        if error == "raise":
            raise type(exc)(msg) from None
        if error == "warn":
            warnings.warn(msg, category=ImportWarning, stacklevel=2)

    return None
