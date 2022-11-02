"""Test string_ops module for processors."""

# pylint: disable=expression-not-assigned

import numpy as np
import pytest

from cyclops.processors.constants import EMPTY_STRING
from cyclops.processors.string_ops import (
    compute_range_avg,
    convert_to_numeric,
    count_occurrences,
    fill_missing_with_nan,
    fix_inequalities,
    is_non_empty_string,
    is_range,
    none_to_empty_string,
    normalize_special_characters,
    remove_text_in_parentheses,
    replace_if_string_match,
    strip_whitespace,
    to_lower,
)


def test_to_lower():
    """Test to_lower fn."""
    assert to_lower("KoBe") == "kobe"


def test_fix_inequalities():
    """Test fix_inequalities fn."""
    assert fix_inequalities("<= 10") == "10"
    assert fix_inequalities("> 10.2") == "10.2"
    assert fix_inequalities(">10.2") == "10.2"


def test_fill_missing_with_nan():
    """Test fill_missing_with_nan fn."""
    assert np.isnan(fill_missing_with_nan(EMPTY_STRING))
    assert fill_missing_with_nan("kobe") == "kobe"


def test_replace_if_string_match():
    """Test replace_if_string_match fn."""
    assert (
        replace_if_string_match("kobe is the greatest of all time", "greatest", "best")
        == "best"
    )
    assert (
        replace_if_string_match("kobe is awesome", "best", "best") == "kobe is awesome"
    )


def test_is_non_empty_string():
    """Test is_non_empty_string fn."""
    assert is_non_empty_string(EMPTY_STRING) is False
    assert is_non_empty_string("kobe") is True


def test_remove_text_in_parentheses():
    """Test remove_text_in_parentheses fn."""
    assert remove_text_in_parentheses("kobe (TM)") == "kobe"
    assert remove_text_in_parentheses("(TM)") == EMPTY_STRING


def test_none_to_empty_string():
    """Test none_to_empty_string fn."""
    assert none_to_empty_string(None) == EMPTY_STRING
    assert none_to_empty_string("kobe") == "kobe"


def test_strip_whitespace():
    """Test strip_whitespace fn."""
    assert strip_whitespace(" kobe time ") == "kobetime"


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
