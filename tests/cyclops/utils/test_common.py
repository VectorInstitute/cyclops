"""Test common utility fns."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cyclops.utils.common import (
    add_years_approximate,
    add_years_exact,
    list_swap,
    to_datetime_format,
    to_list,
    to_list_optional,
    to_timestamp,
)


def test_to_timestamp():
    """Test to_timestamp fn."""
    series = pd.Series([1.2, 2000, 909998888300000])
    timestamp = to_timestamp(series)
    assert timestamp[0].year == 1970
    assert timestamp[2].year == 1970
    assert timestamp[2].day == 11

    array = np.array([1.2, 2000, 909998888300000])
    timestamp = to_timestamp(array)
    assert timestamp[0].year == 1970
    assert timestamp[2].year == 1970
    assert timestamp[2].day == 11

    with pytest.raises(ValueError):
        to_timestamp("donkey")


def test_add_years_approximate():
    """Test add_years_approximate fn."""
    datetime_series = pd.Series([datetime(2022, 11, 3, hour=13)])
    years_series = pd.Series([5])
    result_series = add_years_approximate(datetime_series, years_series)
    assert result_series[0].year == 2027


def test_add_years_exact():
    """Test add_years_exact fn."""
    datetime_series = pd.Series([datetime(2022, 11, 3, hour=13)])
    years_series = pd.Series([5])
    with pytest.warns() as _:
        result_series = add_years_exact(datetime_series, years_series)
    assert result_series[0].year == 2027


def test_to_list():
    """Test to_list fn."""
    assert to_list("kobe") == ["kobe"]
    assert to_list(np.array([1, 2])) == [1, 2]
    assert to_list({1, 2, 3}) == [1, 2, 3]


def test_to_list_optional():
    """Test to_list_optional fn."""
    assert to_list_optional("kobe") == ["kobe"]
    assert to_list_optional(np.array([1, 2])) == [1, 2]
    assert to_list_optional(None) is None
    assert to_list_optional([1]) == [1]
    assert to_list_optional(None, True) == []


def test_to_datetime_format():
    """Test to_datetime_format fn."""
    date = to_datetime_format("1992-11-07")
    assert date.year == 1992
    assert date.month == 11
    assert date.day == 7


def test_list_swap():
    """Test list_swap fn."""
    with pytest.raises(ValueError):
        list_swap([0, 1, 2, 8], 4, 1)
    with pytest.raises(ValueError):
        list_swap([0, 1, 2, 8], 1, 4)

    assert list_swap([0, 1, 2, 8], 0, 1) == [1, 0, 2, 8]
