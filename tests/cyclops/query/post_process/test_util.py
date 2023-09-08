"""Tests for post-processing functions in the query package."""

import pandas as pd

from cyclops.query.post_process.util import event_time_between


def test_event_time_between():
    """Test event_time_between fn."""
    admit_ts = pd.Series(
        [
            pd.Timestamp(year=2017, month=1, day=1, hour=12),
            pd.Timestamp(year=2017, month=1, day=1, hour=12),
        ],
    )
    discharge_ts = pd.Series(
        [
            pd.Timestamp(year=2017, month=1, day=7, hour=12),
            pd.Timestamp(year=2018, month=1, day=9, hour=12),
        ],
    )
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=2),
        admit_ts,
        discharge_ts,
    )
    assert is_between[0]
    assert is_between[1]
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=7, hour=12),
        admit_ts,
        discharge_ts,
    )
    assert not is_between[0]
    assert is_between[1]
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=7, hour=12),
        admit_ts,
        discharge_ts,
        discharge_inclusive=True,
    )
    assert is_between[0]
    assert is_between[1]
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=1, hour=12),
        admit_ts,
        discharge_ts,
        admit_inclusive=False,
    )
    assert not is_between[0]
    assert not is_between[1]
