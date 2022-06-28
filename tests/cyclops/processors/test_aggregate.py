"""Test aggregation functions."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.aggregate import Aggregator
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
)
from cyclops.processors.constants import MEAN


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
        [1, "eventA", 10, date1, date2, "wash"],
        [2, "eventA", 19, date1, date2, "clean"],
        [2, "eventA", 11, date1, date4, "dog"],
        [2, "eventA", 18, date1, date4, "pet"],
        [2, "eventA", 12, date1, date3, "store"],
        [2, "eventA", 16, date1, date3, "kobe"],
        [2, "eventA", np.nan, date1, date3, "bryant"],
        [2, "eventB", 13, date1, date3, "trump"],
        [2, "eventB", np.nan, date1, date3, "tie"],
    ]
    columns = [
        ENCOUNTER_ID,
        EVENT_NAME,
        EVENT_VALUE,
        ADMIT_TIMESTAMP,
        EVENT_TIMESTAMP,
        "some_str_col",
    ]

    return pd.DataFrame(data, columns=columns)


def test_aggregate_events(  # pylint: disable=redefined-outer-name
    test_aggregate_events_input,
):
    """Test aggregate_events function."""
    # Test initializations of Aggregator.
    aggregator = Aggregator(
        aggfuncs={EVENT_VALUE: MEAN},
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        bucket_size=1,
        window_duration=20,
    )
    res = aggregator(test_aggregate_events_input)
    print(res)

    # assert res["timestep"][3] == 14
    # assert res["event_value"][1] == 14.5
    # assert res["event_value"][2] == 19
    # assert res["count"][4] == 2
    # assert res["null_fraction"][4] == 0.5
    # assert res["some_str_col"][3] == "store"

    # assert aggregator.meta[TIMESTEP_START_TIMESTAMP][TIMESTEP_START_TIMESTAMP][1][
    #    18
    # ] == datetime(2022, 11, 4, 7)
    # assert aggregator.meta[TIMESTEP_START_TIMESTAMP][TIMESTEP_START_TIMESTAMP][2][
    #    4
    # ] == datetime(2022, 11, 3, 17)
