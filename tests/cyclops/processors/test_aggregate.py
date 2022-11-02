"""Test aggregation functions."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas import Timestamp

from cyclops.processors.aggregate import AGGFUNCS, Aggregator
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    RESTRICT_TIMESTAMP,
    START_TIMESTAMP,
    START_TIMESTEP,
    STOP_TIMESTAMP,
    TIMESTEP,
)
from cyclops.processors.constants import MEAN, MEDIAN

DATE1 = datetime(2022, 11, 3, hour=13)
DATE2 = datetime(2022, 11, 3, hour=14)
DATE3 = datetime(2022, 11, 4, hour=3)
DATE4 = datetime(2022, 11, 4, hour=13)


@pytest.fixture
def test_input():
    """Create a test events input."""
    data = [
        [1, "eventA", 10, DATE2, "wash"],
        [2, "eventA", 19, DATE2, "clean"],
        [2, "eventA", 11, DATE4, "dog"],
        [2, "eventA", 18, DATE4, "pet"],
        [2, "eventA", 12, DATE3, "store"],
        [2, "eventA", 16, DATE3, "kobe"],
        [2, "eventA", np.nan, DATE3, "bryant"],
        [2, "eventB", 13, DATE3, "trump"],
        [2, "eventB", np.nan, DATE3, "tie"],
        [2, "eventB", np.nan, DATE1, "aaa"],
    ]
    columns = [
        ENCOUNTER_ID,
        EVENT_NAME,
        EVENT_VALUE,
        EVENT_TIMESTAMP,
        "some_str_col",
    ]

    data = pd.DataFrame(data, columns=columns)

    window_start_data = [
        [2, DATE2],
    ]
    window_start = pd.DataFrame(
        window_start_data, columns=[ENCOUNTER_ID, RESTRICT_TIMESTAMP]
    )
    window_start = window_start.set_index(ENCOUNTER_ID)

    window_stop_data = [
        [2, DATE3],
    ]
    window_stop = pd.DataFrame(
        window_stop_data, columns=[ENCOUNTER_ID, RESTRICT_TIMESTAMP]
    )
    window_stop = window_stop.set_index(ENCOUNTER_ID)

    return data, window_start, window_stop


def test_aggregate_events(  # pylint: disable=redefined-outer-name
    test_input,
):
    """Test aggregation function."""
    data, _, _ = test_input

    # Test initializations of Aggregator.
    aggregator = Aggregator(
        aggfuncs={EVENT_VALUE: MEAN},
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
        agg_meta_for=EVENT_VALUE,
    )
    res = aggregator(data)

    assert res.index.names == [ENCOUNTER_ID, EVENT_NAME, TIMESTEP]
    assert res.loc[(2, "eventA", 1)][EVENT_VALUE] == 19
    assert res.loc[(2, "eventA", 14)][EVENT_VALUE] == 14
    assert np.isnan(res.loc[(2, "eventB", 0)][EVENT_VALUE])
    assert res.loc[(2, "eventB", 14)][EVENT_VALUE] == 13

    assert res.loc[(2, "eventB", 0)][START_TIMESTEP] == DATE1


def test_aggregate_window_duration(  # pylint: disable=redefined-outer-name
    test_input,
):
    """Test aggregation window duration functionality."""
    data, _, _ = test_input

    aggregator = Aggregator(
        aggfuncs={EVENT_VALUE: MEAN},
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
        window_duration=12,
    )

    res = aggregator(data)
    res = res.reset_index()
    assert (res[TIMESTEP] < 2).all()


def test_aggregate_start_stop_windows(  # pylint: disable=redefined-outer-name
    test_input,
):
    """Test manually providing start/stop time windows."""
    data, window_start_time, window_stop_time = test_input

    aggregator = Aggregator(
        aggfuncs={EVENT_VALUE: MEAN},
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
    )

    res = aggregator(
        data,
        window_start_time=window_start_time,
        window_stop_time=window_stop_time,
    )

    assert res.loc[(2, "eventA", 0)][START_TIMESTEP] == DATE2

    res = res.reset_index().set_index(ENCOUNTER_ID)
    assert res.loc[2][TIMESTEP].max() <= 13

    aggregator = Aggregator(
        aggfuncs={EVENT_VALUE: MEAN},
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
        window_duration=10,
    )
    try:
        res = aggregator(data, window_stop_time=window_stop_time)
        raise ValueError(
            """Should have raised an error that window_duration cannot be set when
            window_stop_time is specified."""
        )
    except ValueError:
        pass


def test_aggregate_strings(  # pylint: disable=redefined-outer-name
    test_input,
):
    """Test that using aggregation strings is equivalent to inputting the functions."""
    data, _, _ = test_input

    for string, func in AGGFUNCS.items():
        aggregator_str = Aggregator(
            aggfuncs={EVENT_VALUE: string},
            timestamp_col=EVENT_TIMESTAMP,
            time_by=ENCOUNTER_ID,
            agg_by=[ENCOUNTER_ID, EVENT_NAME],
            timestep_size=1,
            window_duration=20,
        )

        aggregator_fn = Aggregator(
            aggfuncs={EVENT_VALUE: func},
            timestamp_col=EVENT_TIMESTAMP,
            time_by=ENCOUNTER_ID,
            agg_by=[ENCOUNTER_ID, EVENT_NAME],
            timestep_size=1,
            window_duration=20,
        )

        assert aggregator_str(data).equals(aggregator_fn(data))

    try:
        aggregator_str = Aggregator(
            aggfuncs={EVENT_VALUE: "shubaluba"},
            timestamp_col=EVENT_TIMESTAMP,
            time_by=ENCOUNTER_ID,
            agg_by=[ENCOUNTER_ID, EVENT_NAME],
            timestep_size=1,
            window_duration=20,
        )
    except ValueError:
        pass


def test_aggregate_multiple(  # pylint: disable=redefined-outer-name
    test_input,
):
    """Test with multiple columns over which to aggregate."""
    data, _, _ = test_input

    data["event_value2"] = 2 * data[EVENT_VALUE]

    aggregator = Aggregator(
        aggfuncs={
            EVENT_VALUE: MEAN,
            "event_value2": MEAN,
        },
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
        window_duration=20,
    )

    res = aggregator(data)

    res = res.reset_index()
    assert res["event_value2"].equals(res[EVENT_VALUE] * 2)


def test_aggregate_one_group_outlier():
    """Test very specific one group outlier case (currently still broken).

    If only one group in the agg_by and the timesteps form a range, e.g., 0-N,
    then the agg_by columns and TIMESTEP are dropped and an index range is returned.

    An example of this setup can be seen below, and currently it is still broken.

    """
    data = [
        [
            0,
            "eventA",
            10.0,
            Timestamp("2022-11-03 14:00:00"),
            "wash",
            Timestamp("2022-11-03 14:00:00"),
            Timestamp("2022-11-03 14:00:00"),
            0,
        ],
        [
            0,
            "eventA",
            10.0,
            Timestamp("2022-11-03 14:00:00"),
            "wash",
            Timestamp("2022-11-03 14:00:00"),
            Timestamp("2022-11-03 14:00:00"),
            1,
        ],
        [
            0,
            "eventA",
            10.0,
            Timestamp("2022-11-03 14:00:00"),
            "wash",
            Timestamp("2022-11-03 14:00:00"),
            Timestamp("2022-11-03 14:00:00"),
            2,
        ],
    ]
    columns = [
        ENCOUNTER_ID,
        EVENT_NAME,
        EVENT_VALUE,
        EVENT_TIMESTAMP,
        "some_str_col",
        START_TIMESTAMP,
        STOP_TIMESTAMP,
        TIMESTEP,
    ]

    data = pd.DataFrame(data, columns=columns)

    aggregator = Aggregator(
        aggfuncs={EVENT_VALUE: MEAN},
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
    )

    _ = data.groupby(aggregator.agg_by, sort=False, group_keys=False).apply(
        aggregator._compute_aggregation  # pylint: disable=protected-access
    )


def test_vectorization(  # pylint: disable=redefined-outer-name
    test_input,
):
    """Test vectorization of aggregated data."""
    data, _, _ = test_input

    data["event_value2"] = 2 * data[EVENT_VALUE]
    data["event_value3"] = 3 * data[EVENT_VALUE]

    aggregator = Aggregator(
        aggfuncs={
            EVENT_VALUE: MEAN,
            "event_value2": MEAN,
            "event_value3": MEDIAN,
        },
        timestamp_col=EVENT_TIMESTAMP,
        time_by=ENCOUNTER_ID,
        agg_by=[ENCOUNTER_ID, EVENT_NAME],
        timestep_size=1,
        window_duration=15,
    )

    aggregated = aggregator(data)

    vectorized_obj = aggregator.vectorize(aggregated)
    vectorized, indexes = vectorized_obj.data, vectorized_obj.indexes

    agg_col_index, encounter_id_index, event_name_index, timestep_index = indexes

    assert set(list(encounter_id_index)) == set([1, 2])
    assert set(list(event_name_index)) == set(["eventA", "eventB"])
    assert set(list(timestep_index)) == set(range(15))

    assert vectorized.shape == (3, 2, 2, 15)
    assert np.array_equal(
        vectorized[list(agg_col_index).index(EVENT_VALUE)] * 2,
        vectorized[list(agg_col_index).index("event_value2")],
        equal_nan=True,
    )
    assert np.array_equal(
        vectorized[list(agg_col_index).index(EVENT_VALUE)] * 3,
        vectorized[list(agg_col_index).index("event_value3")],
        equal_nan=True,
    )
