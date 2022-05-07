"""Test functions that map diagnoses codes to ICD trajectories (features/target)."""

import pandas as pd

from cyclops.processors.column_names import DIAGNOSIS_CODE, ENCOUNTER_ID
from cyclops.processors.constants import EMPTY_STRING
from cyclops.processors.diagnoses import (
    get_alphabet,
    get_icd_category,
    get_numeric,
    group_diagnosis_codes_to_trajectories,
    insert_decimal,
)


def test_insert_decimal():
    """Test insert_decimal fn."""
    assert insert_decimal("232", 1) == "2.32"


def test_get_numeric():
    """Test get_numeric fn."""
    assert get_numeric("M55") == "55"


def test_get_alphabet():
    """Test get_alphabet fn."""
    assert get_alphabet("M55") == "M"


def test_get_icd_category():
    """Test get_icd_category fn."""
    trajectories = {
        "cat": ("E00", "E10"),
        "dog": ("M00", "M05"),
    }
    assert get_icd_category("M03.3.2", trajectories) == "M00_M05"
    assert get_icd_category("M3.3.2", trajectories) == EMPTY_STRING
    assert get_icd_category("E09.8", trajectories) == "E00_E10"


def test_group_diagnosis_codes_to_trajectories():
    """Test group_diagnosis_codes_to_trajectories fn."""
    input_ = pd.DataFrame(index=[0, 1, 2], columns=[ENCOUNTER_ID, DIAGNOSIS_CODE])
    trajectories = {
        "cat": ("E00", "E10"),
        "dog": ("M00", "M05"),
    }
    input_.loc[0] = ["cat", "E09.8"]
    input_.loc[1] = ["cat", "M09.8"]
    input_.loc[2] = ["dog", "M03.8"]

    trajectory_features = group_diagnosis_codes_to_trajectories(input_, trajectories)
    assert all(trajectory_features["E00_E10"] == [1, 0])
    assert all(trajectory_features["M00_M05"] == [0, 1])
