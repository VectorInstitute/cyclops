"""Test OMOP query API."""

import pytest

from cyclops.query.omop import OMOPQuerier


@pytest.mark.integration_test
def test_omop_querier():
    """Test OMOPQuerier."""
    synthea = OMOPQuerier("cdm_synthea10", ["database=synthea_integration_test"])
    visits = synthea.visit_occurrence().run()
    persons = synthea.person().run()
    observations = synthea.observation().run()
    measurements = synthea.measurement().run()
    visit_details = synthea.visit_detail().run()
    assert len(visits) == 4115
    assert len(visit_details) == 4115
    assert len(persons) == 109
    assert len(observations) == 16169
    assert len(measurements) == 19373
