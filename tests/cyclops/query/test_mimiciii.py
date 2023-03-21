"""Test MIMICIII query API."""

import pytest

from cyclops.query import MIMICIIIQuerier


@pytest.mark.integration_test
def test_mimiciii_querier():
    """Test MIMICIIIQuerier."""
    querier = MIMICIIIQuerier()
    custom_tables = querier.list_custom_tables()
    assert "diagnoses" in custom_tables
    assert "labevents" in custom_tables
    assert "chartevents" in custom_tables
