"""Test clean module."""


import pandas as pd
import pytest

from cyclops.process.clean import normalize_events


ENCOUNTER_ID = "enc_id"
EVENT_NAME = "event_name"
EVENT_VALUE = "event_value"
EVENT_VALUE_UNIT = "event_value_unit"


@pytest.fixture()
def test_event_data_normalized():
    """Create event data test input with normalized event values."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4],
        columns=[ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, EVENT_VALUE_UNIT, "dummy"],
    )
    input_.loc[0] = ["sheep", "test-a", 0.3, "unit-a", "hi"]
    input_.loc[1] = ["cat", "test-b", 1.4, "unit-b", "hellow"]
    input_.loc[2] = ["cat", "test-A", 1.2, "Unit-a", "meow"]
    input_.loc[4] = ["dog", "test-c", 0, "unit-c", "wuff"]
    return input_


@pytest.fixture()
def test_event_data_unnormalized():
    """Create event data test input with unnormalized event values."""
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


def test_normalize_events(
    test_event_data_unnormalized,
    test_event_data_normalized,
):
    """Test normalize_events fn."""
    normalized_events = normalize_events(
        test_event_data_normalized,
        event_name_col=EVENT_NAME,
        event_value_col=EVENT_VALUE,
        event_value_unit_col=EVENT_VALUE_UNIT,
    )

    assert len(normalized_events[EVENT_NAME].unique()) == 3
    assert len(normalized_events[EVENT_VALUE_UNIT].unique()) == 3
    assert "test-a" in list(normalized_events[EVENT_NAME])
    assert "unit-a" in list(normalized_events[EVENT_VALUE_UNIT])

    normalized_events = normalize_events(
        test_event_data_unnormalized,
        event_name_col=EVENT_NAME,
        event_value_col=EVENT_VALUE,
        event_value_unit_col=EVENT_VALUE_UNIT,
    )
    assert normalized_events[EVENT_VALUE][0] == 1.0
    assert normalized_events[EVENT_VALUE][1] == 1.4
    assert normalized_events[EVENT_VALUE][2] == 1.2
