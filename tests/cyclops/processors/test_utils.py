"""Test processor utility functions."""

import pandas as pd

from cyclops.processors.utils import (
    check_must_have_columns,
    gather_columns,
    is_timeseries_data,
)


def test_is_timeseries_data():
    """Test is_timeseries_data fn."""
    test_input = pd.DataFrame(index=[0, 1])
    assert is_timeseries_data(test_input) is False

    test_input = pd.DataFrame(index=pd.MultiIndex.from_product([[0, 1], range(2)]))
    assert is_timeseries_data(test_input) is True


def test_check_must_have_columns():
    """Test check_must_have_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert check_must_have_columns(test_input, ["A", "B", "C"]) is True
    assert check_must_have_columns(test_input, ["A", "C"]) is True
    assert check_must_have_columns(test_input, ["D", "C"]) is False


def test_gather_columns():
    """Test gather_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert "B" not in gather_columns(test_input, ["A", "C"])
