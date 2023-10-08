"""Test eICU query API."""

import pytest

from cyclops.query import EICUQuerier


@pytest.mark.integration_test()
def test_eicu_querier():
    """Test EICUQuerier on eICU-CRD."""
    querier = EICUQuerier(
        database="eicu",
        user="postgres",
        password="pwd",
    )

    patients = querier.eicu_crd.patient().run(limit=10)
    assert len(patients) == 10
    assert "age" in patients

    diagnoses = querier.eicu_crd.diagnosis().run(limit=10)
    assert len(diagnoses) == 10
    assert "diagnosisstring" in diagnoses

    vital_periods = querier.eicu_crd.vitalperiodic().run(limit=10)
    assert "heartrate" in vital_periods

    vital_aperiodic = querier.eicu_crd.vitalaperiodic().run(limit=10)
    assert "pvri" in vital_aperiodic
