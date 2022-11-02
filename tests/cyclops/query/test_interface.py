"""Test functions for interface module in query package."""

import os
import shutil
from typing import Generator
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
    query_interface.run()

    query_interface.data = test_data
    path = os.path.join("test_save", "test_features.parquet")
    query_interface.save(path)
    loaded_data = pd.read_parquet(path)
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")
    query_interface.clear_data()
    assert not query_interface.data

    query_interface.data = None
    query_interface.save(path)
    query_interface.save(path, file_format="csv")
    with pytest.raises(ValueError):
        query_interface.save(path, file_format="donkey")

    generator = query_interface.run_in_grouped_batches(id_col="donkey", batch_size=10)
    assert isinstance(generator, Generator)

    query_interface.save_in_grouped_batches("test_save", id_col="donkey", batch_size=10)


@patch("cyclops.orm.Database")
@patch("sqlalchemy.sql.selectable.Subquery")
def test_query_interface_processed(
    database, query, test_data  # pylint: disable=redefined-outer-name
):
    """Test QueryInterface."""
    # Identity fn for post-processing.
    query_interface = QueryInterfaceProcessed(database, query, lambda x: x)
    query_interface.run()

    query_interface.data = test_data
    path = os.path.join("test_save", "test_features.parquet")
    query_interface.save(path)
    loaded_data = pd.read_parquet(path)
    assert loaded_data.equals(test_data)
    shutil.rmtree("test_save")
    query_interface.clear_data()
    assert not query_interface.data
