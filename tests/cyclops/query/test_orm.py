"""Test cyclops.query.orm module."""

import os

import pytest
import pandas as pd

from cyclops.query import OMOPQuerier


@pytest.mark.integration_test
def test_omop_querier():
    """Test ORM using OMOPQuerier."""
    querier = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")
    assert querier is not None
    db = querier._db
    visits_query = querier.visit_occurrence().query
    db.save_query_to_csv(visits_query, "visits.csv")
    visits_df = pd.read_csv("visits.csv")
    assert len(visits_df) == 4115
    os.remove("visits.csv")

    db.save_query_to_parquet(visits_query, "visits.parquet")
    visits_df = pd.read_parquet("visits.parquet")
    assert len(visits_df) == 4115
    os.remove("visits.parquet")

    assert len(list(querier._db.tables(querier.schema_name))) == 44
