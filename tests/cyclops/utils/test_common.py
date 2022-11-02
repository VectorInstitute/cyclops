"""Test common utility fns."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cyclops.utils.common import (
    add_years_approximate,
    add_years_exact,
    append_if_missing,
    array_series_conversion,
    is_one_dimensional,
    list_swap,
    print_dict,
    to_datetime_format,
    to_list,
    to_list_optional,
)


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


def test_print_dict(capfd):
    """Test print_dict fn."""
    test_dict = {"blackbird": "single"}
    print_dict(test_dict)
    out, _ = capfd.readouterr()
    assert out == "{'blackbird': 'single'}\n"

    with pytest.raises(ValueError):
        print_dict(test_dict, limit=-1)

    test_dict = {"a": 1, "b": 2, "c": 3}
    print_dict(test_dict, limit=2)
    out, _ = capfd.readouterr()
    assert out == "{'a': 1, 'b': 2}\n"


def test_list_swap():
    """Test list_swap fn."""
    with pytest.raises(ValueError):
        list_swap([0, 1, 2, 8], 4, 1)
    with pytest.raises(ValueError):
        list_swap([0, 1, 2, 8], 1, 4)

    assert list_swap([0, 1, 2, 8], 0, 1) == [1, 0, 2, 8]


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

    with pytest.raises(ValueError):
        array_series_conversion(to="donkey")
    with pytest.raises(ValueError):
        array_series_conversion(to="series", out_to="donkey")

    @array_series_conversion(to="series", out_to="none")
    def test9(*datas):
        assert all(isinstance(data, pd.Series) for data in datas)
        return datas

    array_ret, series_ret = test9(array, series)
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

    @array_series_conversion(to="series")
    def test7(*datas):
        return (datas[0], datas[1], datas[0])

    with pytest.raises(ValueError):
        _ = test7(array, series)

    @array_series_conversion(to="series")
    def test8(*datas):
        return datas[0]

    with pytest.raises(ValueError):
        _ = test8(array, array)


def test_is_one_dimensional():
    """Test is_one_dimensional fn."""
    assert is_one_dimensional(np.array([1, 3, 4])) is True
    assert is_one_dimensional(np.array([[1], [3], [4]]), raise_error=False) is False
    with pytest.raises(ValueError):
        is_one_dimensional(np.array([[1], [3], [4]]))
