"""Test common utility fns."""

import numpy as np

from cyclops.utils.common import (
    append_if_missing,
    to_datetime_format,
    to_list,
    to_list_optional,
)


def test_to_list():
    """Test to_list fn."""
    assert to_list("kobe") == ["kobe"]
    assert to_list(np.array([1, 2])) == [1, 2]


def test_to_list_optional():
    """Test to_list_optional fn."""
    assert to_list_optional("kobe") == ["kobe"]
    assert to_list_optional(np.array([1, 2])) == [1, 2]
    assert to_list_optional(None) is None
    assert to_list_optional([1]) == [1]


def test_to_datetime_format():
    """Test to_datetime_format fn."""
    date = to_datetime_format("1992-11-07")
    assert date.year == 1992
    assert date.month == 11
    assert date.day == 7


def test_append_if_missing():
    """Test append_if_missing fn."""
    out_list = append_if_missing([3], [3, 4, 5])
    assert out_list == [3, 4, 5]
    out_list = append_if_missing([], ["a"])
    assert out_list == ["a"]
    out_list = append_if_missing(["a", "b"], ["a", "b"])
    assert out_list == ["a", "b"]
