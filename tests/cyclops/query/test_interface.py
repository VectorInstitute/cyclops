"""Test functions for interface module in query package."""

import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed


@pytest.fixture
def test_data():
    """Dummy dataframe for testing."""
    return pd.DataFrame([[1, "a", 1], [5.1, "b", 0]], columns=["col1", "col2", "col3"])


@patch("cyclops.orm.Database")
@patch("sqlalchemy.sql.selectable.Subquery")
def test_query_interface(
    database, query, test_data  # pylint: disable=redefined-outer-name
):
    """Test QueryInterface."""
    query_interface = QueryInterface(database, query)
    with pytest.raises(ValueError):
        _ = query_interface.run(filter_columns=["a"], filter_recognised=True)

    query_interface.data = test_data
    query_interface.save("test_save", "test_features")
    loaded_data = pd.read_parquet(os.path.join("test_save", "query_test_features.gzip"))
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")
    query_interface.clear_data()
    assert not query_interface.data

    # Doesn't test the actual filtering of the query attributes, just the call.
    _ = query_interface.run(filter_columns=["a"])
    _ = query_interface.run(filter_recognised=True)


@patch("cyclops.orm.Database")
@patch("sqlalchemy.sql.selectable.Subquery")
def test_query_interface_processed(
    database, query, test_data  # pylint: disable=redefined-outer-name
):
    """Test QueryInterface."""
    # Identity fn for post-processing.
    query_interface = QueryInterfaceProcessed(database, query, lambda x: x)
    with pytest.raises(ValueError):
        _ = query_interface.run(filter_columns=["a"], filter_recognised=True)

    query_interface.data = test_data
    query_interface.save("test_save", "test_features")
    loaded_data = pd.read_parquet(os.path.join("test_save", "query_test_features.gzip"))
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")
    query_interface.clear_data()
    assert not query_interface.data

    # Doesn't test the actual filtering of the query attributes, just the call.
    _ = query_interface.run(filter_columns=["a"])
    _ = query_interface.run(filter_recognised=True)
