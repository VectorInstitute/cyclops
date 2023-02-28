"""Test eICU query API."""

import pytest

from cyclops.query import EICUQuerier

# pylint: disable=no-member


@pytest.mark.integration_test
def test_eicu_querier():
    """Test EICUQuerier on eICU-CRD."""
    eicu = EICUQuerier(database="eicu")
    patients = eicu.patient().run(limit=10)
    assert len(patients) == 10
    assert "age" in patients

    diagnoses = eicu.diagnosis().run(limit=10)
    assert len(diagnoses) == 10
    assert "diagnosisstring" in diagnoses

    vital_periods = eicu.vitalperiodic().run(limit=10)
    assert "heartrate" in vital_periods

    vital_aperiodic = eicu.vitalaperiodic().run(limit=10)
    assert "pvri" in vital_aperiodic
