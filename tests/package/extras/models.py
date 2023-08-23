"""Test import of subpackages with base cyclops + models extra install."""

import pytest


def test_import_cyclops():
    """Test import of cyclops."""
    import cyclops

    assert cyclops.__name__ == "cyclops"

    import cyclops.data as data

    assert data.__name__ == "cyclops.data"

    import cyclops.evaluate as evaluate

    assert evaluate.__name__ == "cyclops.evaluate"

    import cyclops.models as models

    assert models.__name__ == "cyclops.models"

    import cyclops.monitor as monitor

    assert monitor.__name__ == "cyclops.monitor"

    with pytest.raises(ImportError):
        import cyclops.query
    with pytest.raises(ImportError):
        import cyclops.report
