"""Test functions that process static features."""

import numpy as np
import pandas as pd
import pytest

from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.processors.statics import compute_statics, infer_statics


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
    with pytest.raises(ValueError):
        _ = infer_statics(test_statics_input, "donkey")
    static_columns = infer_statics(test_statics_input)
    assert set(static_columns) == set([ENCOUNTER_ID, "B", "E"])


def test_compute_statics(  # pylint: disable=redefined-outer-name
    test_statics_input,
):
    """Test aggregate_statics function."""
    statics = compute_statics(test_statics_input)

    assert statics["B"].loc["cat"] == 2
    assert statics["B"].loc["donkey"] == 3
    assert statics["B"].loc["sheep"] == 6
    assert statics["E"].loc["cat"] == "c"
    assert np.isnan(statics["E"].loc["donkey"])
    assert statics["E"].loc["sheep"] == "s"
