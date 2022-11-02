"""Test query API util functions."""

import pytest
from sqlalchemy import Table, column, select
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops.query.util import (
    DBTable,
    _to_select,
    _to_subquery,
    drop_columns,
    equals,
    filter_columns,
    get_column,
    get_column_names,
    get_columns,
    has_columns,
    not_equals,
    process_column,
    process_elem,
    process_list,
    rename_columns,
    reorder_columns,
    table_params_to_type,
    trim_columns,
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
    with pytest.raises(ValueError):
        table_params_to_type(int)


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
    filtered = filter_columns(test_table, ["a", "c", "d"])
    assert get_column_names(filtered) == ["a", "c"]


def test_has_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test has_columns fn."""
    assert not has_columns(test_table, ["a", "d"])
    assert has_columns(test_table, ["a", "b"])
    with pytest.raises(ValueError):
        has_columns(test_table, ["a", "d"], raise_error=True)


def test_drop_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test drop_columns fn."""
    after_drop = drop_columns(test_table, ["a"])
    assert get_column_names(after_drop) == ["b", "c"]


def test_rename_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test rename_columns fn."""
    after_rename = rename_columns(test_table, {"a": "apple", "b": "ball"})
    assert get_column_names(after_rename) == ["apple", "ball", "c"]


def test_reorder_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test reorder_columns fn."""
    with pytest.raises(ValueError):
        reorder_columns(test_table, ["ball", "c", "a"])
    with pytest.raises(ValueError):
        reorder_columns(test_table, ["c", "a"])
    after_reorder = reorder_columns(test_table, ["b", "c", "a"])
    assert get_column_names(after_reorder) == ["b", "c", "a"]


def test_trim_columns(test_table):  # pylint: disable=redefined-outer-name
    """Test apply_to_columns fn."""
    after_trim = trim_columns(test_table, ["a"], ["apple"])
    assert get_column_names(after_trim) == ["a", "b", "c", "apple"]


def test_process_elem():
    """Test process_elem fn."""
    assert process_elem("Test", lower=True) == "test"
    assert process_elem("Test ", lower=True, trim=True) == "test"
    assert process_elem("1", to_int=True) == 1
    assert process_elem("1.2", to_float=True) == 1.2
    assert process_elem(1, to_bool=True) is True
    assert process_elem(0, to_bool=True) is False


def test_process_list():
    """Test process_list fn."""
    assert process_list([1, 2, 3, 0], to_bool=True) == [True, True, True, False]


def test_process_column():
    """Test process_column fn."""
    test_col = column("a")
    processed_col = process_column(test_col, to_int=True)
    assert str(processed_col) == "CAST(a AS INTEGER)"
    processed_col = process_column(test_col, to_float=True)
    assert str(processed_col) == "CAST(a AS FLOAT)"
    processed_col = process_column(test_col, to_str=True)
    assert str(processed_col) == "CAST(a AS VARCHAR)"
    processed_col = process_column(test_col, to_bool=True)
    assert str(processed_col) == "CAST(a AS BOOLEAN)"
    processed_col = process_column(test_col, to_date=True)
    assert str(processed_col) == "CAST(a AS DATE)"
    processed_col = process_column(test_col, to_timestamp=True)
    assert str(processed_col) == "CAST(a AS TIMESTAMP)"
    test_col.type = "VARCHAR"
    processed_col = process_column(test_col, lower=True, trim=True)
    assert str(processed_col) == "trim(lower(a))"


def test_equals():
    """Test equals fn."""
    test_col = column("a")
    assert str(equals(test_col, "bat")) == "a = :a_1"


def test_not_equals():
    """Test not_equals fn."""
    test_col = column("a")
    assert str(not_equals(test_col, "bat")) == "a != :a_1"
