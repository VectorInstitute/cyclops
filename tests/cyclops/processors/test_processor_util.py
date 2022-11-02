"""Test processor utility functions."""

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.util import (
    assert_has_columns,
    create_indicator_variables,
    gather_columns,
    has_columns,
    has_range_index,
    is_timestamp_series,
    to_range_index,
)


def test_create_indicator_variables():
    """Test create_indicator_variables fn."""
    features = pd.DataFrame([[np.nan, 1], [3, np.nan]], columns=["A", "B"])
    indicator_features = create_indicator_variables(features)
    assert (indicator_features.columns == ["A_indicator", "B_indicator"]).all()
    assert (indicator_features["A_indicator"] == pd.Series([0, 1])).all()
    indicator_features = create_indicator_variables(features, columns=["A"])
    assert (indicator_features.columns == ["A_indicator"]).all()
    assert (indicator_features["A_indicator"] == pd.Series([0, 1])).all()


def test_has_columns():
    """Test has_columns fn."""
    test_input = pd.DataFrame(index=[0, 1], columns=["A", "B", "C"])
    assert has_columns(test_input, ["A", "B", "C"]) is True
    assert has_columns(test_input, ["A", "C"]) is True
    assert has_columns(test_input, ["D", "C"]) is False
    with pytest.raises(ValueError):
        has_columns(test_input, ["D", "C"], raise_error=True)
    with pytest.raises(ValueError):
        has_columns(test_input, ["B", "C"], exactly=True, raise_error=True)


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


def test_to_range_index():
    """Test to_range_index fn."""
    test_df = pd.DataFrame(index=[1, 3], columns=["A", "B"])
    test_df.index.name = "index"
    test_df_with_range_index = to_range_index(test_df)
    test_df = pd.DataFrame(columns=["A", "B"])
    assert to_range_index(test_df).equals(test_df)
    assert has_range_index(test_df_with_range_index)


def test_is_timestamp_series():
    """Test is_timestamp_series fn."""
    assert is_timestamp_series(pd.to_datetime(["2000-01-21 01:30:00"]))
    with pytest.raises(ValueError):
        is_timestamp_series(pd.Series(["a", "b"]), raise_error=True)
