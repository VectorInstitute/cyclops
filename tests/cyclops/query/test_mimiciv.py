"""Test MIMICIV-2.0 query API."""

import pytest

from cyclops.query import MIMICIVQuerier


@pytest.mark.integration_test()
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

    custom_tables = querier.list_custom_tables()
    assert "patients" in custom_tables
    assert "diagnoses" in custom_tables
    assert "labevents" in custom_tables
    assert "chartevents" in custom_tables

    with pytest.raises(AttributeError):
        querier.get_table("invalid_schema", "invalid_table")
