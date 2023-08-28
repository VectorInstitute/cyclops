"""Test import of subpackages with base cyclops + report extra install."""

import pytest


def test_import_cyclops():
    """Test import of cyclops."""
    import cyclops

    assert cyclops.__name__ == "cyclops"

    from cyclops import data

    assert data.__name__ == "cyclops.data"

    from cyclops import report

    assert report.__name__ == "cyclops.report"

    with pytest.raises(ImportError):
        import cyclops.evaluate
    with pytest.raises(ImportError):
        import cyclops.models
    with pytest.raises(ImportError):
        import cyclops.query
    with pytest.raises(ImportError):
        import cyclops.monitor
