"""Test aggregation functions."""

from datetime import datetime

import pandas as pd
import pytest

from cyclops.processors.aggregate import filter_upto_window
from cyclops.processors.column_names import ADMIT_TIMESTAMP, EVENT_TIMESTAMP


@pytest.fixture
def test_input():
    """Create a test input."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4], columns=["A", "B", "C", EVENT_TIMESTAMP, ADMIT_TIMESTAMP]
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


def test_filter_upto_window(test_input):  # pylint: disable=redefined-outer-name
    """Test filter_upto_window fn."""
    filtered_df = filter_upto_window(test_input)
    assert "dog" not in filtered_df["A"]
    filtered_df = filter_upto_window(test_input, start_at_admission=True)
    assert "dog" not in filtered_df["A"] and "3" not in filtered_df["B"]
    filtered_df = filter_upto_window(
        test_input, start_window_ts=datetime(2022, 11, 6, 12, 13)
    )
    assert len(filtered_df) == 1 and "dog" == filtered_df["A"].item()
    with pytest.raises(ValueError):
        filtered_df = filter_upto_window(
            test_input,
            start_window_ts=datetime(2022, 11, 6, 12, 13),
            start_at_admission=True,
        )
