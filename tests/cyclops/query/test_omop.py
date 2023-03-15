"""Test OMOP query API."""

import pytest

import cyclops.query.ops as qo
from cyclops.query import OMOPQuerier


@pytest.mark.integration_test
def test_omop_querier_synthea():
    """Test OMOPQuerier on synthea data."""
    synthea = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")
    ops = qo.Sequential(
        [
            qo.ConditionEquals("gender_source_value", "M"),
            qo.Rename({"race_source_value": "race"}),
        ]
    )
    persons_qi = synthea.person(ops=ops)
    visits = synthea.visit_occurrence(
        join=qo.JoinArgs(join_table=persons_qi.query, on="person_id")
    ).run()
    persons = persons_qi.run()
    observations = synthea.observation().run()
    measurements = synthea.measurement().run()
    visit_details = synthea.visit_detail().run()
    providers = synthea.provider().run()  # pylint: disable=no-member
    conditions = synthea.condition_occurrence().run()
    assert len(persons) == 54
    assert len(visits) == 1620
    assert len(visit_details) == 4115
    assert len(observations) == 16169
    assert len(measurements) == 19373
    assert len(providers) == 212
    assert len(conditions) == 1363


@pytest.mark.integration_test
def test_omop_querier_mimiciii():
    """Test OMOPQuerier on MIMICIII data."""
    mimic = OMOPQuerier("omop", database="mimiciii")
    visits = mimic.visit_occurrence().run()
    assert len(visits) == 58976
