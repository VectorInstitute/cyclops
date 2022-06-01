"""Test events processing module."""

from datetime import datetime

import pandas as pd
import pytest

from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
)
from cyclops.processors.events import (
    combine_events,
    convert_to_events,
    normalise_events,
)


@pytest.fixture
def test_event_data_normalised():
    """Create event data test input with normalised event values."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4],
        columns=[ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, EVENT_VALUE_UNIT, "dummy"],
    )
    input_.loc[0] = ["sheep", "test-a", 0.3, "unit-a", "hi"]
    input_.loc[1] = ["cat", "test-b", 1.4, "unit-b", "hellow"]
    input_.loc[2] = ["cat", "test-A", 1.2, "Unit-a", "meow"]
    input_.loc[4] = ["dog", "test-c", 0, "unit-c", "wuff"]
    return input_


def test_combine_events():
    """Test combine_events fn."""
    test_input1 = pd.DataFrame(
        columns=[ENCOUNTER_ID, EVENT_TIMESTAMP, EVENT_NAME, EVENT_VALUE],
        index=[0],
    )
    test_input2 = pd.DataFrame(
        columns=[ENCOUNTER_ID, EVENT_TIMESTAMP, EVENT_NAME, EVENT_VALUE],
        index=[0, 1],
    )
    test_input1.loc[0] = [12, datetime(2022, 11, 3, 12, 13), "eventA", 1.2]
    test_input2.loc[0] = [14, datetime(2022, 11, 3, 1, 13), "eventB", 11.2]
    test_input2.loc[1] = [12, datetime(2022, 11, 4, 12, 13), "eventA", 111.2]

    events = combine_events([test_input1, test_input2])
    assert len(events) == 3
    assert events.loc[2][EVENT_NAME] == "eventA"

    events = combine_events([test_input1])
    assert events.equals(test_input1)


def test_convert_to_events():
    """Test convert_to_events fn."""
    test_input = pd.DataFrame(columns=[ENCOUNTER_ID, "test_ts"], index=[0, 1, 2])
    test_input.loc[0] = [12, datetime(2022, 11, 3, 12, 13)]
    test_input.loc[1] = [11, datetime(2022, 11, 3, 19, 13)]
    test_input.loc[2] = [1, datetime(2022, 11, 2, 1, 1)]
    events = convert_to_events(test_input, event_name="test", timestamp_col="test_ts")
    assert len(events) == 3
    assert events.loc[2][ENCOUNTER_ID] == 1
    assert events.loc[1][EVENT_TIMESTAMP] == datetime(2022, 11, 3, 19, 13)

    test_input = pd.DataFrame(
        columns=[ENCOUNTER_ID, "test_ts", "test_value"], index=[0, 1, 2]
    )
    test_input.loc[0] = [12, datetime(2022, 11, 3, 12, 13), 1.2]
    test_input.loc[1] = [11, datetime(2022, 11, 3, 19, 13), 2.34]
    test_input.loc[2] = [1, datetime(2022, 11, 2, 1, 1), 11]
    events = convert_to_events(
        test_input, event_name="test", timestamp_col="test_ts", value_col="test_value"
    )
    assert len(events) == 3
    assert events.loc[2][ENCOUNTER_ID] == 1
    assert events.loc[1][EVENT_TIMESTAMP] == datetime(2022, 11, 3, 19, 13)
    assert events.loc[2][EVENT_VALUE][0] == 11
    assert events.loc[2][EVENT_NAME] == "test"


@pytest.fixture
def test_event_data_unnormalised():
    """Create event data test input with unnormalised event values."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4],
        columns=[ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, EVENT_VALUE_UNIT],
    )
    input_.loc[0] = [
        "sheep",
        "test-a",
        "positive",
        "unit-a",
    ]
    input_.loc[1] = [
        "cat",
        "test-b",
        "<1.4",
        "unit-b",
    ]
    input_.loc[2] = [
        "cat",
        "test-A",
        "1.2 (L)",
        "Unit-a",
    ]
    input_.loc[4] = [
        "dog",
        "test-c",
        "",
        "unit-c",
    ]
    return input_


def test_normalise_events(  # pylint: disable=redefined-outer-name
    test_event_data_unnormalised, test_event_data_normalised
):
    """Test normalise_events fn."""
    normalised_events = normalise_events(test_event_data_normalised)

    assert len(normalised_events[EVENT_NAME].unique()) == 3
    assert len(normalised_events[EVENT_VALUE_UNIT].unique()) == 3
    assert "test-a" in list(normalised_events[EVENT_NAME])
    assert "unit-a" in list(normalised_events[EVENT_VALUE_UNIT])

    normalised_events = normalise_events(test_event_data_unnormalised)
    assert normalised_events[EVENT_VALUE][0] == 1.0
    assert normalised_events[EVENT_VALUE][1] == 1.4
    assert normalised_events[EVENT_VALUE][2] == 1.2
