"""Test base dataset querier, using OMOPQuerier as an example."""

import pytest

from cyclops.query import OMOPQuerier


@pytest.mark.integration_test()
def test_dataset_querier():
    """Test base querier methods using OMOPQuerier."""
    querier = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")
    assert len(querier.list_tables()) == 69
    assert len(querier.list_schemas()) == 4
    assert len(querier.list_tables(schema_name="cdm_synthea10")) == 43
    visit_occrrence_columns = querier.list_columns("cdm_synthea10", "visit_occurrence")
    assert len(visit_occrrence_columns) == 17
    assert "visit_occurrence_id" in visit_occrrence_columns
