"""Test string_ops module for processors."""

# pylint: disable=expression-not-assigned

import numpy as np
import pytest

from cyclops.processors.string_ops import (
    count_occurrences,
    normalize_special_characters,
    convert_to_numeric,
    is_range,
    compute_range_avg,
)


def test_count_occurrences():
    """Test count_occurrences fn."""
    test_case1 = ["kobe", "jordan", "magic", "durant", "kobe", "magic", "kobe"]
    test_case2 = [1, 1, 2, 4, 9]

    counts = count_occurrences(test_case1)
    assert counts[0][0] == "kobe"
    assert counts[1][0] == "magic"
    counts = count_occurrences(test_case2)
    assert counts[0][0] == 1


def test_normalize_special_characters():
    """Test normalize_special_characters fn."""
    test_input = "test% result+ & g/mol #2"
    normalized = normalize_special_characters(test_input)
    assert normalized == "test_percent_result_plus_and_g_per_mol_number_2"


def test_is_range():
    """Test is_range fn."""
    assert is_range("1 to 2") is True
    assert is_range("5 - 10") is True
    assert is_range("1 2") is False
    assert is_range("1- 2") is False
    assert is_range("1-2") is False


def test_convert_to_numeric():
    """Test convert_to_numeric fn."""
    assert convert_to_numeric(None) is np.nan
    assert convert_to_numeric(np.nan) is np.nan
    with pytest.raises(TypeError):
        convert_to_numeric(10)
    assert convert_to_numeric("5 - 10") == 7.5


def test_compute_range_avg():
    """Test compute_range_avg fn."""
    compute_range_avg("5 - 10") == 7.5
    compute_range_avg("2 - 4") == 3
