"""Test MIMICIV-2.0 query API."""

import pytest

from cyclops.query import MIMICIVQuerier


@pytest.mark.integration_test
def test_mimiciv_querier():
    """Test MIMICQuerier on MIMICIV-2.0."""
    mimic = MIMICIVQuerier()
    patients = mimic.patients().run(limit=10)
    assert len(patients) == 10
    assert "anchor_year_difference" in patients

    diagnoses = mimic.diagnoses().run(limit=10)
    assert len(diagnoses) == 10
    assert "long_title" in diagnoses

    care_units = mimic.care_units().run(limit=10)
    assert "admit" in care_units
    assert "discharge" in care_units

    lab_events = mimic.labevents().run(limit=10)
    assert "category" in lab_events

    chart_events = mimic.chartevents().run(limit=10)
    assert "value" in chart_events
    assert "category" in chart_events

    tables = mimic.list_tables()
    assert "patients" in tables
    assert "diagnoses" in tables
    assert "labevents" in tables
    assert "chartevents" in tables
    assert "pharmacy" in tables

    with pytest.raises(ValueError):
        mimic.get_table("invalid_table")
