"""Test pandas util fns."""

from datetime import datetime

import pandas as pd
import pytest

from cyclops.utils.pandas import add_years_approximate, add_years_exact


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
