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
    get_columns,
    has_columns,
)


@pytest.fixture
def test_table():
    """Test table input."""
    return select(column("a"), column("b"), column("c"))


def test__to_subquery():
    """Test _to_subquery fn."""
    assert isinstance(_to_subquery(select().subquery()), Subquery)
    assert isinstance(_to_subquery(select()), Subquery)
    assert isinstance(_to_subquery(Table()), Subquery)
    assert isinstance(_to_subquery(DBTable("a", Table())), Subquery)
    with pytest.raises(TypeError):
        _to_subquery("a")


def test__to_select():
    """Test _to_select fn."""
    assert isinstance(_to_select(select().subquery()), Select)
    assert isinstance(_to_select(select()), Select)
    assert isinstance(_to_select(Table()), Select)
    assert isinstance(_to_select(DBTable("a", Table())), Select)
    with pytest.raises(TypeError):
        _to_select("a")


def test_get_column(test_table):  # pylint: disable=redefined-outer-name
    """Test get_column fn."""
    assert str(get_column(test_table, "a")) == "anon_1.a"
    with pytest.raises(ValueError):
        get_column(select(column("a")), "b")


def test_get_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test get_columns fn."""
    cols = get_columns(test_table, "c")
    cols = [str(col) for col in cols]
    assert cols == ["anon_1.c"]
    with pytest.raises(ValueError):
        get_column(select(column("a")), "b")


def test_get_column_names(test_table):  # pylint: disable=redefined-outer-name
    """Test get_column_names fn."""
    assert get_column_names(test_table) == ["a", "b", "c"]


def test_filter_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test filter_columns fn."""
    filtered = filter_columns(test_table, ["a", "c"])
    assert get_column_names(filtered) == ["a", "c"]


def test_has_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test has_columns fn."""
    assert not has_columns(test_table, ["a", "d"])
    assert has_columns(test_table, ["a", "b"])
    with pytest.raises(ValueError):
        has_columns(test_table, ["a", "d"], raise_error=True)
