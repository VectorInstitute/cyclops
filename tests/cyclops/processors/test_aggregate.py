"""Test aggregation functions."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.aggregate import (
    Aggregator,
    aggregate_values_in_bucket,
    filter_upto_window,
    gather_event_features,
    gather_events_into_single_bucket,
    gather_statics,
    get_earliest_ts_encounter,
    infer_statics,
)
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
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
def test_statics_input():
    """Input dataframe to test infer_statics fn."""
    input_ = pd.DataFrame(
        index=list(range(4)),
        columns=[ENCOUNTER_ID, "B", "C", "D", "E"],
    )
    input_.loc[0] = ["cat", np.nan, "0", 0, "c"]
    input_.loc[1] = ["cat", 2, "1", 1, np.nan]
    input_.loc[2] = ["sheep", 6, "0", 0, "s"]
    input_.loc[3] = ["sheep", np.nan, "0", 0, "s"]
    input_.loc[4] = ["donkey", 3, "1", 1, np.nan]

    return input_


def test_infer_statics(  # pylint: disable=redefined-outer-name
    test_statics_input,
):
    """Test infer_statics fn."""
    static_columns = infer_statics(test_statics_input)
    assert set(static_columns) == set([ENCOUNTER_ID, "B", "E"])


def test_gather_statics(  # pylint: disable=redefined-outer-name
    test_statics_input,
):
    """Test gather_statics function."""
    statics = gather_statics(test_statics_input)

    assert statics["B"].loc["cat"] == 2
    assert statics["B"].loc["donkey"] == 3
    assert statics["B"].loc["sheep"] == 6
    assert statics["E"].loc["cat"] == "c"
    assert np.isnan(statics["E"].loc["donkey"])
    assert statics["E"].loc["sheep"] == "s"


def test_filter_upto_window(test_input):  # pylint: disable=redefined-outer-name
    """Test filter_upto_window fn."""
    filtered_df = filter_upto_window(test_input)
    assert "dog" not in filtered_df[ENCOUNTER_ID]
    filtered_df = filter_upto_window(test_input, start_at_admission=True)
    assert "dog" not in filtered_df[ENCOUNTER_ID] and "3" not in filtered_df["B"]
    filtered_df = filter_upto_window(
        test_input, start_window_ts=datetime(2022, 11, 6, 12, 13)
    )
    assert len(filtered_df) == 1 and "dog" == filtered_df[ENCOUNTER_ID].item()
    with pytest.raises(ValueError):
        filtered_df = filter_upto_window(
            test_input,
            start_window_ts=datetime(2022, 11, 6, 12, 13),
            start_at_admission=True,
        )


def test_aggregate_values_in_bucket():
    """Test aggregate_values_in_bucket function."""
    arr = np.array([-1, 2, 5, -20, 100])
    series = pd.Series(arr)
    assert aggregate_values_in_bucket(series, strategy=MEAN) == arr.mean()
    assert aggregate_values_in_bucket(series, strategy=MEDIAN) == np.median(arr)
    with pytest.raises(NotImplementedError):
        aggregate_values_in_bucket(series, strategy="donkey")


def test_gather_events_into_single_bucket(  # pylint: disable=redefined-outer-name
    test_events_input,
):
    """Test gather_events_into_single_bucket function."""
    res = gather_events_into_single_bucket(test_events_input, MEAN)
    assert res.loc[1]["eventA"] == 10.0
    assert np.isnan(res.loc[1]["eventB"])
    assert res.loc[2]["eventA"] == 13.0
    assert res.loc[2]["eventB"] == 13.0

    res = gather_events_into_single_bucket(test_events_input, MEDIAN)
    assert res.loc[1]["eventA"] == 10.0
    assert np.isnan(res.loc[1]["eventB"])
    assert res.loc[2]["eventA"] == 12.0
    assert res.loc[2]["eventB"] == 13.0


def test_get_earliest_ts_encounter(  # pylint: disable=redefined-outer-name
    test_events_input,
):
    """Test get_earliest_ts_encounter fn."""
    earliest_ts = get_earliest_ts_encounter(test_events_input)
    assert earliest_ts[1] == datetime(2022, 11, 3, hour=14)
    assert earliest_ts[2] == datetime(2022, 11, 3, hour=14)


def test_gather_event_features(  # pylint: disable=redefined-outer-name
    test_events_input,
):
    """Test gather_event_features function."""
    agg = Aggregator(strategy=MEAN, bucket_size=4, window=24, start_at_admission=True)
    res = gather_event_features(test_events_input, agg)

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

    agg = Aggregator(strategy=MEAN, bucket_size=4, window=4)
    res = gather_event_features(test_events_input, agg)
    assert all(a == b for a, b in zip(list(res.index), [1, 2]))
    assert res["eventA"].equals(pd.Series([10, 11], index=[1, 2]))
