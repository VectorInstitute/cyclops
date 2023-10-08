"""Test cyclops.query.orm module."""

import os

import pandas as pd
import pytest

from cyclops.query import OMOPQuerier


@pytest.mark.integration_test()
def test_omop_querier():
    """Test ORM using OMOPQuerier."""
    querier = OMOPQuerier(
        database="synthea_integration_test",
        schema_name="cdm_synthea10",
        user="postgres",
        password="pwd",
    )
    assert querier is not None
    db_ = querier.db
    visits_query = querier.visit_occurrence().query
    db_.save_query_to_csv(visits_query, "visits.csv")
    visits_df = pd.read_csv("visits.csv")
    assert len(visits_df) == 4320
    os.remove("visits.csv")

    db_.save_query_to_parquet(visits_query, "visits.parquet")
    visits_df = pd.read_parquet("visits.parquet")
    assert len(visits_df) == 4320
    os.remove("visits.parquet")
