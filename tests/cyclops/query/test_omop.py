"""Test OMOP query API."""

import pytest

from cyclops.query.omop import OMOPQuerier


@pytest.mark.integration_test
def test_omop_querier():
    """Test OMOPQuerier."""
    synthea = OMOPQuerier("cdm_synthea10", ["database=synthea"])
    visits = synthea.visit_occurrence().run()
    assert len(visits) == 4856
