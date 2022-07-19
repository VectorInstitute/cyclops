"""Test functions that map diagnoses codes to ICD trajectories (features/target)."""

from cyclops.processors.constants import EMPTY_STRING
from cyclops.processors.diagnoses import (
    get_alphabet,
    get_icd_category,
    get_numeric,
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
