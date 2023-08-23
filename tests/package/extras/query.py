"""Test import of subpackages with base cyclops + query extra install."""

import pytest


def test_import_cyclops():
    """Test import of cyclops."""
    import cyclops

    assert cyclops.__name__ == "cyclops"

    import cyclops.data as data

    assert data.__name__ == "cyclops.data"

    import cyclops.query as query

    assert query.__name__ == "cyclops.query"

    with pytest.raises(ImportError):
        import cyclops.evaluate
    with pytest.raises(ImportError):
        import cyclops.models
    with pytest.raises(ImportError):
        import cyclops.monitor
    with pytest.raises(ImportError):
        import cyclops.report
