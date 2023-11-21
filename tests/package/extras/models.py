"""Test import of subpackages with base cyclops + models extra install."""



def test_import_cyclops():
    """Test import of cyclops."""
    import cyclops

    assert cyclops.__name__ == "cyclops"

    from cyclops import data

    assert data.__name__ == "cyclops.data"

    from cyclops import evaluate

    assert evaluate.__name__ == "cyclops.evaluate"

    from cyclops import models

    assert models.__name__ == "cyclops.models"

    from cyclops import monitor

    assert monitor.__name__ == "cyclops.monitor"
