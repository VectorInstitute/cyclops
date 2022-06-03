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
    TIMESTEP_START_TIMESTAMP,
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
    date4 = datetime(2022, 11, 3, hour=13)
    data = [
        [1, "eventA", 10, date1, date2],
        [2, "eventA", 19, date1, date2],
        [2, "eventA", 11, date1, date4],
        [2, "eventA", 18, date1, date4],
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


def test_compute_start_of_window(test_input):  # pylint: disable=redefined-outer-name
    """Test compute_start_of_window fn."""
    aggregator = Aggregator()
    start_time = aggregator.compute_start_of_window(test_input)
    assert (
        start_time.loc[start_time[ENCOUNTER_ID] == "cat"][RESTRICT_TIMESTAMP]
        == datetime(2022, 11, 3, 8, 5)
    ).all()
    aggregator = Aggregator(start_at_admission=True)
    start_time = aggregator.compute_start_of_window(test_input)
    assert (
        start_time.loc[start_time[ENCOUNTER_ID] == "cat"][RESTRICT_TIMESTAMP]
        == datetime(2022, 11, 3, 21, 13)
    ).all()
    start_window_ts = pd.DataFrame(
        [["sheep", datetime(2022, 11, 3, 12, 13)]],
        columns=[ENCOUNTER_ID, RESTRICT_TIMESTAMP],
    )
    aggregator = Aggregator(start_window_ts=start_window_ts)
    start_time = aggregator.compute_start_of_window(test_input)
    assert (
        start_time.loc[start_time[ENCOUNTER_ID] == "sheep"][RESTRICT_TIMESTAMP]
        == datetime(2022, 11, 3, 12, 13)
    ).all()


def test_aggregate_events_single_timestep_case(  # pylint: disable=redefined-outer-name
    test_aggregate_events_input,
):
    """Test aggregate_events function for single timestep case."""
    aggregtor = Aggregator(aggfunc=MEAN, bucket_size=4, window=4)
    res = aggregtor(test_aggregate_events_input)
    assert res[EVENT_NAME][0] == "eventA"
    assert res[EVENT_NAME][1] == "eventA"
    assert res["count"][1] == 3

    aggregtor = Aggregator(aggfunc=MEDIAN, bucket_size=4, window=40)
    res = aggregtor(test_aggregate_events_input)
    assert res[EVENT_VALUE][1] == 18
    assert res[EVENT_NAME][2] == "eventA"
    assert res["null_fraction"][3] == 0.5


def test_get_earliest_ts_encounter(  # pylint: disable=redefined-outer-name
    test_events_input,
):
    """Test get_earliest_ts_encounter fn."""
    earliest_ts = get_earliest_ts_encounter(test_events_input)
    assert (
        earliest_ts.loc[earliest_ts[ENCOUNTER_ID] == 1][EVENT_TIMESTAMP]
        == datetime(2022, 11, 3, hour=14)
    ).all()
    assert (
        earliest_ts.loc[earliest_ts[ENCOUNTER_ID] == 2][EVENT_TIMESTAMP]
        == datetime(2022, 11, 3, hour=14)
    ).all()


def test_aggregate_events(  # pylint: disable=redefined-outer-name
    test_aggregate_events_input,
):
    """Test aggregate_events function."""
    # Test initializations of Aggregator.
    with pytest.raises(NotImplementedError):
        _ = Aggregator(
            aggfunc="donkey", bucket_size=4, window=20, start_at_admission=True
        )
    _ = Aggregator(aggfunc=np.mean, bucket_size=2, window=20, start_at_admission=True)

    with pytest.raises(ValueError):
        _ = Aggregator(
            start_window_ts=test_restrict_events_by_timestamp_start_input,
            start_at_admission=True,
        )
    with pytest.raises(ValueError):
        _ = Aggregator(
            stop_window_ts=test_restrict_events_by_timestamp_stop_input, window=12
        )

    aggregator = Aggregator(
        aggfunc=MEAN, bucket_size=1, window=20, start_at_admission=True
    )
    res = aggregator(test_aggregate_events_input)

    assert res["timestep"][3] == 14
    assert res["event_value"][1] == 14.5
    assert res["event_value"][2] == 19
    assert res["count"][4] == 2
    assert res["null_fraction"][4] == 0.5

    assert aggregator.meta[TIMESTEP_START_TIMESTAMP][TIMESTEP_START_TIMESTAMP][1][
        18
    ] == datetime(2022, 11, 4, 7)
    assert aggregator.meta[TIMESTEP_START_TIMESTAMP][TIMESTEP_START_TIMESTAMP][2][
        4
    ] == datetime(2022, 11, 3, 17)

    _ = Aggregator(
        aggfunc=MEAN,
        bucket_size=1,
        window=None,
        start_window_ts=test_restrict_events_by_timestamp_start_input,
        stop_window_ts=test_restrict_events_by_timestamp_stop_input,
    )


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
    assert list(res.index) == [0, 4, 5, 6, 7, 8]

    res = restrict_events_by_timestamp(
        test_aggregate_events_input,
        stop=test_restrict_events_by_timestamp_stop_input,
    )
    assert list(res.index) == [0, 1, 2, 3]

    res = restrict_events_by_timestamp(test_aggregate_events_input)
    assert res.equals(test_aggregate_events_input)
