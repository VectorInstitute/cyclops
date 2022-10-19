"""Test low-level query API processing functions."""

import pytest
from sqlalchemy import column, select

from cyclops.query.process import (
    QAP,
    ConditionIn,
    ConditionSubstring,
    Drop,
    Rename,
    process_checks,
    process_operations,
)
from cyclops.query.util import process_column


@pytest.fixture
def test_table():
    """Test table input."""
    column_a = process_column(column("a"), to_timestamp=True)
    return select(column_a, column("b"), column("c"))


def test_process_checks(test_table):  # pylint: disable=redefined-outer-name
    """Test process_checks fn."""
    process_checks(test_table, cols=["a"], cols_not_in=["d"], timestamp_cols=["a"])
    with pytest.raises(ValueError):
        process_checks(test_table, cols_not_in=["a"])


def test_process_operations(test_table):  # pylint: disable=redefined-outer-name
    """Test process_operations fn."""
    process_kwargs = {"b_args": "cat", "c_args": ["dog"]}
    operations = [
        (
            ConditionIn,
            ["b", QAP("b_args")],
            {"to_str": True},
        ),
        (ConditionSubstring, ["c", QAP("c_args")], {}),
    ]
    table = process_operations(test_table, operations, process_kwargs)
    query_lines = str(table).splitlines()
    assert query_lines[0] == "SELECT anon_1.a, anon_1.b, anon_1.c "
    assert query_lines[-1] == "WHERE lower(CAST(anon_1.c AS VARCHAR)) LIKE :lower_1"


def test_drop(test_table):  # pylint: disable=redefined-outer-name
    """Test Drop operation."""
    table = Drop("a")(test_table)
    assert str(table).splitlines()[0] == "SELECT anon_1.b, anon_1.c "


def test_rename(test_table):  # pylint: disable=redefined-outer-name
    """Test Rename operation."""
    table = Rename({"a": "d"})(test_table)
    assert str(table).splitlines()[0] == "SELECT anon_1.a AS d, anon_1.b, anon_1.c "
