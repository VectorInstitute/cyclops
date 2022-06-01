"""Test processor utility functions."""

from typing import Optional

import pandas as pd

from cyclops.processors.util import (
    assert_has_columns,
    gather_columns,
    has_columns,
    is_timeseries_data,
)


def test_is_timeseries_data():
    """Test is_timeseries_data fn."""
    test_input = pd.DataFrame(index=[0, 1])
    assert is_timeseries_data(test_input) is False

    test_input = pd.DataFrame(index=pd.MultiIndex.from_product([[0, 1], range(2)]))
    assert is_timeseries_data(test_input) is True


def test_has_columns():
    """Test has_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert has_columns(test_input, ["A", "B", "C"]) is True
    assert has_columns(test_input, ["A", "C"]) is True
    assert has_columns(test_input, ["D", "C"]) is False


def test_assert_has_columns():
    """Test assert_has_columns decorator."""

    @assert_has_columns(
        ["A", "B"],
        None,  # No check on df2
        ["Pizza"],
        df_kwarg=["sauce", "please"],
    )
    def test(  # pylint: disable=too-many-arguments, unused-argument
        df1: pd.DataFrame,
        some_int: int,
        df2: pd.DataFrame,
        some_str: str,
        df3: pd.DataFrame,
        int_keyword: int = None,
        df_kwarg: Optional[pd.DataFrame] = None,
    ) -> None:
        return None

    df1 = pd.DataFrame(columns=["A", "B", "C"])
    some_int = 1
    df2 = pd.DataFrame(columns=["C", "D"])
    some_str = "A"
    df3 = pd.DataFrame(columns=["Pizza", "is", "yummy"])
    int_keyword = 2
    df_kwarg = pd.DataFrame(columns=["Extra", "sauce", "please"])

    # Passing tests
    test(df1, some_int, df2, some_str, df3, int_keyword=int_keyword, df_kwarg=df_kwarg)
    test(df1, some_int, df2, some_str, df3, df_kwarg=df_kwarg)

    # Failing tests
    df1_fail = pd.DataFrame(columns=["A", "C"])
    try:
        test(
            df1_fail,
            some_int,
            df2,
            some_str,
            df3,
            int_keyword=int_keyword,
            df_kwarg=df_kwarg,
        )
        assert False
    except ValueError as error:
        assert "B" in str(error)

    df_kwarg_fail = pd.DataFrame(columns=["hiya"])

    try:
        test(
            df1,
            some_int,
            df2,
            some_str,
            df3,
            int_keyword=int_keyword,
            df_kwarg=df_kwarg_fail,
        )
        assert False
    except ValueError as error:
        assert "sauce" in str(error)


def test_gather_columns():
    """Test gather_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert "B" not in gather_columns(test_input, ["A", "C"])
