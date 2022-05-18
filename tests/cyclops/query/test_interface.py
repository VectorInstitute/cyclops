"""Test functions for interface module in query package."""

import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from cyclops.query.interface import QueryInterface, save_queried_dataframe


def test_save_queried_dataframe():
    """Test save fn."""
    dummy_data = pd.DataFrame(
        [[1, "a", 1], [5.1, "b", 0]], columns=["col1", "col2", "col3"]
    )
    save_queried_dataframe(dummy_data, "test_save", "test_features")
    loaded_data = pd.read_parquet(os.path.join("test_save", "test_features.gzip"))
    assert loaded_data.equals(dummy_data)
    shutil.rmtree("test_save")
    # Does nothing, just prints log warning!
    save_queried_dataframe(None, "test_save", "test_features")


@patch("cyclops.orm.Database")
@patch("sqlalchemy.sql.selectable.Subquery")
def test_query_interface(database, query):
    """Test QueryInterface."""
    query_interface = QueryInterface(database, query)
    with pytest.raises(ValueError):
        _ = query_interface.run(filter_columns=["a"], filter_recognised=True)
