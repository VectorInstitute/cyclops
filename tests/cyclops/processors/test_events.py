"""Test events processing module."""

import pandas as pd
import pytest

from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
)
from cyclops.processors.events import clean_events


@pytest.fixture
def test_event_data_normalized():
    """Create event data test input with normalized event values."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4],
        columns=[ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, EVENT_VALUE_UNIT],
    )
    input_.loc[0] = [
        "sheep",
        "test-a",
        0.3,
        "unit-a",
    ]
    input_.loc[1] = [
        "cat",
        "test-b",
        1.4,
        "unit-b",
    ]
    input_.loc[2] = [
        "cat",
        "test-A",
        1.2,
        "Unit-a",
    ]
    input_.loc[4] = [
        "dog",
        "test-c",
        0,
        "unit-c",
    ]
    return input_


@pytest.fixture
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


def test_clean_events(  # pylint: disable=redefined-outer-name
    test_event_data_unnormalized, test_event_data_normalized
):
    """Test clean_events fn."""
    cleaned_events = clean_events(test_event_data_normalized)

    assert len(cleaned_events[EVENT_NAME].unique()) == 3
    assert len(cleaned_events[EVENT_VALUE_UNIT].unique()) == 3
    assert "test-a" in list(cleaned_events[EVENT_NAME])
    assert "unit-a" in list(cleaned_events[EVENT_VALUE_UNIT])

    cleaned_events = clean_events(test_event_data_unnormalized)
    assert cleaned_events[EVENT_VALUE][0] == 1.0
    assert cleaned_events[EVENT_VALUE][1] == 1.4
    assert cleaned_events[EVENT_VALUE][2] == 1.2
