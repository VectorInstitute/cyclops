"""Test OMOP query API."""

import pytest

import cyclops.query.ops as qo
from cyclops.query import OMOPQuerier


@pytest.mark.integration_test()
def test_omop_querier_synthea():
    """Test OMOPQuerier on synthea data."""
    querier = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")
    ops = qo.Sequential(
        qo.ConditionEquals("gender_source_value", "M"),
        qo.Rename({"race_source_value": "race"}),
    )
    persons = querier.person()
    persons = persons.ops(ops)
    visits = querier.visit_occurrence()
    visits = visits.join(persons, "person_id").run()
    persons = persons.run()
    observations = querier.observation().run()
    measurements = querier.measurement().run()
    visit_details = querier.visit_detail().run()
    providers = querier.cdm_synthea10.provider().run()
    conditions = querier.cdm_synthea10.condition_occurrence().run()
    assert len(persons) == 54
    assert len(visits) == 1798
    assert len(visit_details) == 4320
    assert len(observations) == 17202
    assert len(measurements) == 19994
    assert len(providers) == 212
    assert len(conditions) == 1419


@pytest.mark.integration_test()
def test_omop_querier_mimiciii():
    """Test OMOPQuerier on MIMICIII data."""
    querier = OMOPQuerier("omop", database="mimiciii")
    visits = querier.visit_occurrence().run()
    assert len(visits) == 58976
