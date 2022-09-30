"""Test query API util functions."""

import pytest
from sqlalchemy import Table, column, select
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops.query.util import (
    DBTable,
    _to_select,
    _to_subquery,
    filter_columns,
    get_column,
    get_column_names,
)


def test__to_subquery():
    """Test _to_subquery fn."""
    assert isinstance(_to_subquery(select().subquery()), Subquery)
    assert isinstance(_to_subquery(select()), Subquery)
    assert isinstance(_to_subquery(Table()), Subquery)
    assert isinstance(_to_subquery(DBTable("a", Table())), Subquery)


def test__to_select():
    """Test _to_select fn."""
    assert isinstance(_to_select(select().subquery()), Select)
    assert isinstance(_to_select(select()), Select)
    assert isinstance(_to_select(Table()), Select)
    assert isinstance(_to_select(DBTable("a", Table())), Select)


def test_get_column():
    """Test get_column fn."""
    assert str(get_column(select(column("a")), "a")) == "anon_1.a"
    with pytest.raises(ValueError):
        get_column(select(column("a")), "b")


def test_get_column_names():
    """Test get_column_names fn."""
    assert get_column_names(select(column("a"), column("b"))) == ["a", "b"]


def test_filter_columns():
    """Test filter_columns fn."""
    test_table = select(column("a"), column("b"))
    filtered = filter_columns(test_table, "a")
    assert get_column_names(filtered) == ["a"]
