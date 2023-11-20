"""Utilities for handling optional dependencies."""
import importlib
import importlib.util
import logging
import warnings
from types import ModuleType
from typing import Literal, Optional

from cyclops.utils.log import setup_logging


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="WARN", logger=LOGGER)


def import_optional_module(
    name: str,
    error: Literal["raise", "warn", "ignore"] = "raise",
) -> Optional[ModuleType]:
    """Import an optional module.

    Parameters
    ----------
    name : str
        The name of the module to import.
    error : ErrorOption, optional
        How to handle errors. One of:
        - "raise": raise an error if the module cannot be imported.
        - "warn": raise a warning if the module cannot be imported.
        - "ignore": ignore the missing module and return `None`.

    Returns
    -------
    Optional[ModuleType]
        The imported module, if it exists. Otherwise, `None`.

    """
    if error not in ("raise", "warn", "ignore"):
        raise ValueError(
            "Expected `error` to be one of 'raise', 'warn, or 'ignore', "
            f"but got {error}.",
        )

    try:
        return importlib.import_module(name)
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
