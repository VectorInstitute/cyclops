"""Test MIMICIV-2.0 query API."""

import pytest

from cyclops.query import MIMICIVQuerier


@pytest.mark.integration_test
def test_mimiciv_querier():
    """Test MIMICQuerier on MIMICIV-2.0."""
    querier = MIMICIVQuerier()
    patients = querier.patients().run(limit=10)
    assert len(patients) == 10
    assert "anchor_year_difference" in patients

    diagnoses = querier.diagnoses().run(limit=10)
    assert len(diagnoses) == 10
    assert "long_title" in diagnoses

    care_units = querier.care_units().run(limit=10)
    assert "admit" in care_units
    assert "discharge" in care_units

    lab_events = querier.labevents().run(limit=10)
    assert "category" in lab_events

    chart_events = querier.chartevents().run(limit=10)
    assert "value" in chart_events
    assert "category" in chart_events

    tables = querier.list_tables()
    assert "patients" in tables
    assert "diagnoses" in tables
    assert "labevents" in tables
    assert "chartevents" in tables
    assert "pharmacy" in tables

    with pytest.raises(ValueError):
        querier.get_table("invalid_table")
