"""Test optional import utilities."""

import pytest

from cyclops.utils.optional import import_optional_module


class TestImportOptionalModule:
    """Test importing optional modules."""

    def test_import_valid_module(self):
        """Test importing a valid module."""
        module = import_optional_module("math")
        assert module is not None
        import math

        assert module == math

    def test_import_nonexistent_module_ignore(self):
        """Test importing a non-existent module with `error='ignore'`."""
        module = import_optional_module("nonexistent_module", error="ignore")
        assert module is None

    def test_import_nonexistent_module_warn(self):
        """Test importing a non-existent module with `error='warn'`."""
        with pytest.warns(ImportWarning):
            import_optional_module("nonexistent_module", error="warn")

    def test_import_nonexistent_module_raise(self):
        """Test importing a non-existent module with `error='raise'`."""
        with pytest.raises(ModuleNotFoundError):
            import_optional_module("nonexistent_module", error="raise")

    def test_invalid_error_option(self):
        """Test importing a valid module with an invalid error option."""
        with pytest.raises(ValueError):
            import_optional_module("math", error="invalid_option")  # type: ignore
