"""Test common utility fns."""

import numpy as np
import pandas as pd

from cyclops.utils.common import (
    append_if_missing,
    array_series_conversion,
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
    out_list = append_if_missing(["b"], ["a"], to_start=True)
    assert out_list == ["a", "b"]


def test_array_series_conversion():
    """Test array_series_conversion not including out_to='back'."""
    array = np.array([1, np.nan, 2])
    series = pd.Series([1, np.nan, 2])

    # Test single return

    @array_series_conversion(to="array", out_to="array")
    def test1(data):
        assert isinstance(data, np.ndarray)
        return data

    ret = test1(array)
    assert isinstance(ret, np.ndarray)

    ret = test1(series)
    assert isinstance(ret, np.ndarray)

    @array_series_conversion(to="series", out_to="array")
    def test2(data):
        assert isinstance(data, pd.Series)
        return data

    ret = test2(array)
    assert isinstance(ret, np.ndarray)

    ret = test2(series)
    assert isinstance(ret, np.ndarray)

    @array_series_conversion(to="array", out_to="series")
    def test3(data):
        assert isinstance(data, np.ndarray)
        return data

    ret = test3(array)
    assert isinstance(ret, pd.Series)

    ret = test3(series)
    assert isinstance(ret, pd.Series)

    @array_series_conversion(to="series", out_to="series")
    def test4(data):
        assert isinstance(data, pd.Series)
        return data

    ret = test4(array)
    assert isinstance(ret, pd.Series)

    ret = test4(series)
    assert isinstance(ret, pd.Series)

    # Test multiple returns

    @array_series_conversion(to="array", out_to="array")
    def test5(*datas):
        assert all(isinstance(data, np.ndarray) for data in datas)
        return datas

    array_ret, series_ret = test5(array, series)
    assert isinstance(array_ret, np.ndarray)
    assert isinstance(series_ret, np.ndarray)

    @array_series_conversion(to="series", out_to="array")
    def test6(*datas):
        assert all(isinstance(data, pd.Series) for data in datas)
        return datas

    array_ret, series_ret = test6(array, series)
    assert isinstance(array_ret, np.ndarray)
    assert isinstance(series_ret, np.ndarray)

    @array_series_conversion(to="array", out_to="series")
    def test7(*datas):
        assert all(isinstance(data, np.ndarray) for data in datas)
        return datas

    array_ret, series_ret = test7(array, series)
    assert isinstance(array_ret, pd.Series)
    assert isinstance(series_ret, pd.Series)

    @array_series_conversion(to="series", out_to="series")
    def test8(*datas):
        assert all(isinstance(data, pd.Series) for data in datas)
        return datas

    array_ret, series_ret = test8(array, series)
    assert isinstance(array_ret, pd.Series)
    assert isinstance(series_ret, pd.Series)


def test_array_series_conversion_out_to_back():
    """Test array_series_conversion with out_to='back'."""
    array = np.array([1, np.nan, 2])
    series = pd.Series([1, np.nan, 2])

    # Test single return

    # Array to array (no conversion)
    @array_series_conversion(to="array")
    def test1(data):
        assert isinstance(data, np.ndarray)
        return data

    ret = test1(array)
    assert isinstance(ret, np.ndarray)

    # Array to series and back
    @array_series_conversion(to="series")
    def test2(data):
        assert isinstance(data, pd.Series)
        return data

    ret = test2(array)
    assert isinstance(ret, np.ndarray)

    # Series to series (no conversion)
    @array_series_conversion(to="series")
    def test3(data):
        assert isinstance(data, pd.Series)
        return data

    ret = test3(series)
    assert isinstance(ret, pd.Series)

    # Series to to array and back
    @array_series_conversion(to="array")
    def test4(data):
        assert isinstance(data, np.ndarray)
        return data

    ret = test4(series)
    assert isinstance(ret, pd.Series)

    # Test multiple returns

    # To array
    @array_series_conversion(to="array")
    def test5(*datas):
        assert all(isinstance(data, np.ndarray) for data in datas)
        return datas

    array_ret, series_ret = test5(array, series)
    assert isinstance(array_ret, np.ndarray)
    assert isinstance(series_ret, pd.Series)

    # To series
    @array_series_conversion(to="series")
    def test6(*datas):
        assert all(isinstance(data, pd.Series) for data in datas)
        return datas

    array_ret, series_ret = test6(array, series)
    assert isinstance(array_ret, np.ndarray)
    assert isinstance(series_ret, pd.Series)
