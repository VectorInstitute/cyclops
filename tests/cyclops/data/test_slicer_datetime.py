"""Focused datetime slicing tests."""

import pyarrow as pa

from cyclops.data.slicer import filter_datetime


def test_filter_datetime_matches_day_of_month() -> None:
    """Test filtering datetimes by day-of-month."""
    table = pa.table({"dt": ["2020-01-07", "2020-01-14", "2021-01-07"]})

    result = filter_datetime(table, column_name="dt", day=7)

    assert result == [True, False, True]
