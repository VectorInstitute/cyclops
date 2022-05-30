"""Test aggregation functions."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.aggregate import (
    Aggregator,
    get_earliest_ts_encounter,
    restrict_events_by_timestamp,
)
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    RESTRICT_TIMESTAMP,
)
from cyclops.processors.constants import MEAN, MEDIAN


@pytest.fixture
def test_input():
    """Create a test input."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4],
        columns=[ENCOUNTER_ID, "B", "C", EVENT_TIMESTAMP, ADMIT_TIMESTAMP],
    )
    input_.loc[0] = [
        "sheep",
        10,
        "0",
        datetime(2022, 11, 3, 12, 13),
        datetime(2022, 11, 3, 12, 13),
    ]
    input_.loc[1] = [
        "cat",
        2,
        "0",
        datetime(2022, 11, 4, 7, 13),
        datetime(2022, 11, 3, 21, 13),
    ]
    input_.loc[2] = [
        "cat",
        3,
        "1",
        datetime(2022, 11, 3, 8, 5),
        datetime(2022, 11, 3, 21, 13),
    ]
    input_.loc[4] = [
        "dog",
        9.1,
        "0",
        datetime(2022, 11, 6, 18, 1),
        datetime(2022, 11, 3, 12, 13),
    ]
    input_[EVENT_TIMESTAMP] = pd.to_datetime(input_[EVENT_TIMESTAMP])
    input_[ADMIT_TIMESTAMP] = pd.to_datetime(input_[ADMIT_TIMESTAMP])
    return input_


@pytest.fixture
def test_events_input():
    """Create a test events input."""
    date1 = datetime(2022, 11, 3, hour=13)
    date2 = datetime(2022, 11, 3, hour=14)
    date3 = datetime(2022, 11, 4, hour=3)
    data = [
        [1, "eventA", 10, date1, date2],
        [2, "eventA", 11, date1, date2],
        [2, "eventA", 12, date1, date3],
        [2, "eventA", 16, date1, date3],
        [2, "eventB", 13, date1, date3],
    ]

    columns = [ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, ADMIT_TIMESTAMP, EVENT_TIMESTAMP]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def test_aggregate_events_input():
    """Create a test events input."""
    date1 = datetime(2022, 11, 3, hour=13)
    date2 = datetime(2022, 11, 3, hour=14)
    date3 = datetime(2022, 11, 4, hour=3)
    data = [
        [1, "eventA", 10, date1, date2],
        [2, "eventA", 11, date1, date2],
        [2, "eventA", 12, date1, date3],
        [2, "eventA", 16, date1, date3],
        [2, "eventA", np.nan, date1, date3],
        [2, "eventB", 13, date1, date3],
        [2, "eventB", np.nan, date1, date3],
    ]

    columns = [ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, ADMIT_TIMESTAMP, EVENT_TIMESTAMP]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def test_restrict_events_by_timestamp_start_input():
    """Create a restrict_events_by_timestamp start DataFrame input."""
    date1 = datetime(2022, 11, 3, hour=13)
    date3 = datetime(2022, 11, 4, hour=3)
    data = [
        [1, date1],
        [2, date3],
    ]

    columns = [ENCOUNTER_ID, RESTRICT_TIMESTAMP]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def test_restrict_events_by_timestamp_stop_input():
    """Create a restrict_events_by_timestamp stop DataFrame input."""
    date2 = datetime(2022, 11, 3, hour=14)
    data = [[2, date2]]

    columns = [ENCOUNTER_ID, RESTRICT_TIMESTAMP]
    return pd.DataFrame(data, columns=columns)


def test_filter_upto_window(test_input):  # pylint: disable=redefined-outer-name
    """Test filter_upto_window fn."""
    aggregator = Aggregator()
    filtered_df = aggregator.filter_upto_window(test_input)
    assert "dog" not in filtered_df[ENCOUNTER_ID]
    aggregator = Aggregator(start_at_admission=True)
    filtered_df = aggregator.filter_upto_window(test_input)
    assert "dog" not in filtered_df[ENCOUNTER_ID] and "3" not in filtered_df["B"]
    aggregator = Aggregator(start_window_ts=datetime(2022, 11, 6, 12, 13))
    filtered_df = aggregator.filter_upto_window(test_input)
    assert len(filtered_df) == 1 and "dog" == filtered_df[ENCOUNTER_ID].item()


def test_aggregate_events_into_single_bucket(  # pylint: disable=redefined-outer-name
    test_events_input,
):
    """Test aggregate_events_into_single_bucket function."""
    aggregator = Aggregator(aggfunc=MEAN)
    res, _ = aggregator.aggregate_events_into_single_bucket(test_events_input)
    assert res.loc[1]["eventA"] == 10.0
    assert np.isnan(res.loc[1]["eventB"])
    assert res.loc[2]["eventA"] == 13.0
    assert res.loc[2]["eventB"] == 13.0

    aggregator = Aggregator(aggfunc=MEDIAN)
    res, _ = aggregator.aggregate_events_into_single_bucket(test_events_input)
    assert res.loc[1]["eventA"] == 10.0
    assert np.isnan(res.loc[1]["eventB"])
    assert res.loc[2]["eventA"] == 12.0
    assert res.loc[2]["eventB"] == 13.0


def test_aggregate_events_single_timestep_case(  # pylint: disable=redefined-outer-name
    test_aggregate_events_input,
):
    """Test aggregate_events function."""
    aggregtor = Aggregator(aggfunc=MEAN, bucket_size=4, window=4)
    res, _ = aggregtor.aggregate_events(test_aggregate_events_input)
    assert res["eventA"].iloc[0] == 10
    assert res["eventA"].iloc[1] == 11


def test_get_earliest_ts_encounter(  # pylint: disable=redefined-outer-name
    test_events_input,
):
    """Test get_earliest_ts_encounter fn."""
    earliest_ts = get_earliest_ts_encounter(test_events_input)
    assert earliest_ts[1] == datetime(2022, 11, 3, hour=14)
    assert earliest_ts[2] == datetime(2022, 11, 3, hour=14)


def test_aggregate_events(  # pylint: disable=redefined-outer-name
    test_aggregate_events_input,
):
    """Test aggregate_events function."""
    # Test initializations of Aggregator.
    with pytest.raises(NotImplementedError):
        _ = Aggregator(
            aggfunc="donkey", bucket_size=4, window=20, start_at_admission=True
        )
    _ = Aggregator(aggfunc=np.mean, bucket_size=4, window=20, start_at_admission=True)

    with pytest.raises(ValueError):
        _ = Aggregator(
            start_window_ts=datetime(2022, 11, 6, 12, 13), start_at_admission=True
        )

    aggregator = Aggregator(
        aggfunc=MEAN, bucket_size=4, window=20, start_at_admission=True
    )
    res, _ = aggregator.aggregate_events(test_aggregate_events_input)

    assert res.loc[(1, 0)]["eventA"] == 10.0
    assert np.isnan(res.loc[(1, 0)]["eventB"])
    assert res.loc[(2, 0)]["eventA"] == 11.0
    assert np.isnan(res.loc[(2, 0)]["eventB"])
    assert np.isnan(res.loc[(2, 1)]["eventA"])
    assert np.isnan(res.loc[(2, 1)]["eventB"])
    assert np.isnan(res.loc[(2, 2)]["eventA"])
    assert np.isnan(res.loc[(2, 2)]["eventB"])
    assert res.loc[(2, 3)]["eventA"] == 14.0
    assert res.loc[(2, 3)]["eventB"] == 13.0

    # Checks padding past last event
    assert np.isnan(res.loc[(2, 4)]["eventA"])
    assert np.isnan(res.loc[(2, 4)]["eventB"])


def test_restrict_events_by_timestamp(  # pylint: disable=redefined-outer-name
    test_aggregate_events_input,
    test_restrict_events_by_timestamp_start_input,
    test_restrict_events_by_timestamp_stop_input,
):
    """Test restrict_events_by_timestamp function."""
    res = restrict_events_by_timestamp(
        test_aggregate_events_input,
        start=test_restrict_events_by_timestamp_start_input,
    )
    assert list(res.index) == [0, 2, 3, 4, 5, 6]

    res = restrict_events_by_timestamp(
        test_aggregate_events_input,
        stop=test_restrict_events_by_timestamp_stop_input,
    )
    assert list(res.index) == [0, 1]
