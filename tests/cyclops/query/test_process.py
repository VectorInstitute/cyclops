"""Test low-level query API processing functions."""

import pytest
from sqlalchemy import column, select

from cyclops.query.process import process_checks
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
