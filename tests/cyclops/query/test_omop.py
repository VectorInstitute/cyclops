"""Test OMOP query API."""

import pytest

from cyclops.query.omop import VISIT_OCCURRENCE, OMOPQuerier


@pytest.mark.integration_test
def test_omop_querier():
    """Test OMOPQuerier."""
    synthea = OMOPQuerier("cdm_synthea10", ["database=synthea"])
    visits = synthea.get_interface(synthea.get_table(VISIT_OCCURRENCE)).run()
    assert len(visits) == 4856
