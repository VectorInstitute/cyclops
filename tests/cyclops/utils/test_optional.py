"""Test optional import utilities."""

from math import sqrt

import pytest

from cyclops.utils.optional import import_optional_module


def test_import_valid_module():
    """Test importing a valid module."""
    module = import_optional_module("math")
    assert module is not None
    import math

    assert module == math


def test_import_valid_module_attribute():
    """Test importing a valid module attribute."""
    module = import_optional_module("math", attribute="sqrt")
    assert module is not None
    assert module == sqrt


def test_import_nonexistent_module_ignore():
    """Test importing a non-existent module with `error='ignore'`."""
    module = import_optional_module("nonexistent_module", error="ignore")
    assert module is None

    attr = import_optional_module(
        "nonexistent_module",
        attribute="nonexistent_attribute",
        error="ignore",
    )
    assert attr is None


def test_import_nonexistent_module_warn():
    """Test importing a non-existent module with `error='warn'`."""
    with pytest.warns(ImportWarning):
        import_optional_module("nonexistent_module", error="warn")

    with pytest.warns(ImportWarning):
        import_optional_module(
            "nonexistent_module",
            attribute="nonexistent_attribute",
            error="warn",
        )


def test_import_nonexistent_module_raise():
    """Test importing a non-existent module with `error='raise'`."""
    with pytest.raises(ModuleNotFoundError):
        import_optional_module("nonexistent_module", error="raise")

    with pytest.raises(ModuleNotFoundError):
        import_optional_module(
            "nonexistent_module",
            attribute="nonexistent_attribute",
            error="raise",
        )


def test_invalid_error_option():
    """Test importing a valid module with an invalid error option."""
    with pytest.raises(ValueError):
        import_optional_module("math", error="invalid_option")  # type: ignore


def test_import_nonexistent_attribute():
    """Test importing a non-existent attribute."""
    with pytest.raises(AttributeError):
        import_optional_module("math", attribute="nonexistent_attribute")
