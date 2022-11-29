"""Test MIMIC-IV-2.0 query API."""

import pytest

from cyclops.query.mimiciv import MIMICIVQuerier


@pytest.mark.integration_test
def test_omop_querier():
    """Test OMOPQuerier."""
    mimic = MIMICIVQuerier()
    patients = mimic.patients().run()
    assert len(patients) == 315460
