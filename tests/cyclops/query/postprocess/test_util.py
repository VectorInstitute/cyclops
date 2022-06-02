"""Tests for post-processing functions in the query package."""

import numpy as np
import pandas as pd
import pytest

from cyclops.query.postprocess.util import event_time_between, to_timestamp


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


def test_event_time_between():
    """Test event_time_between fn."""
    admit_ts = pd.Series(
        [
            pd.Timestamp(year=2017, month=1, day=1, hour=12),
            pd.Timestamp(year=2017, month=1, day=1, hour=12),
        ]
    )
    discharge_ts = pd.Series(
        [
            pd.Timestamp(year=2017, month=1, day=7, hour=12),
            pd.Timestamp(year=2018, month=1, day=9, hour=12),
        ]
    )
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=2), admit_ts, discharge_ts
    )
    assert is_between[0] and is_between[1]
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=7, hour=12), admit_ts, discharge_ts
    )
    assert not is_between[0] and is_between[1]
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=7, hour=12),
        admit_ts,
        discharge_ts,
        discharge_inclusive=True,
    )
    assert is_between[0] and is_between[1]
    is_between = event_time_between(
        pd.Timestamp(year=2017, month=1, day=1, hour=12),
        admit_ts,
        discharge_ts,
        admit_inclusive=False,
    )
    assert not is_between[0] and not is_between[1]
