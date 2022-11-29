"""String operations used in data extraction."""

import re
from collections import Counter
from typing import Iterable, List, Union

import numpy as np

from cyclops.process.constants import EMPTY_STRING


def fill_missing_with_nan(string: str) -> Union[float, str]:
    """Return NaN if input string is empty.

    Parameters
    ----------
    string: str
        Input string.

    Returns
    -------
    str or float
        NaN if input was empty, else input.

    """
    return np.nan if string == EMPTY_STRING else string


def replace_if_string_match(
    search_string: str, search_terms: str, replace_value: str
) -> str:
    """Replace string with value, if string has matched terms.

    If any of the 'search_terms' are found in the string, then
    it is replaced with the 'replace_value' string.

    Parameters
    ----------
    string: str
        The string to search for possible matches.
    search_terms: str
        String terms to search x1|x2...|xn.
    replace_value: str
        String which will replace if search found any matches.

    Returns
    -------
    str
        The string to replace, 'replace_value' if matches found, else same as input.

    """
    search_string = search_string.lower()
    found = re.search(search_terms, search_string)
    if bool(found):
        return replace_value
    return search_string


def remove_text_in_parentheses(string: str) -> str:
    """Remove text within parentheses.

    e.g. test (T) -> test

    Parameters
    ----------
    string: str
        Input string.

    Returns
    -------
    str
        Output string with text inside including parentheses removed.

    """
    return re.sub(r"\([^)]*\)", "", string).strip()


def fix_inequalities(string: str) -> str:
    """Match result value, remove inequality symbols (<, >, =).

    For e.g.
    10.2, >= 10, < 10, 11 -> 10.2, 10, 10, 11

    Parameters
    ----------
    string: str
        The string to search for possible matches.

    Returns
    -------
    str
        Result value string is matched to regex, symbols removed.

    """
    string = string.lower()
    matches = re.search(
        r"^\s*(<|>)?=?\s*(-?\s*[0-9]+(?P<floating>\.)?(?(floating)[0-9]+|))\s*$",
        string,
    )
    return matches.group(2) if matches else ""


def to_lower(string: str) -> str:
    """Convert string to lowercase letters.

    Parameters
    ----------
    string: str
        Input string.

    Returns
    -------
    str
        Output string in lowercase.

    """
    return string.lower()


def none_to_empty_string(value: Union[None, str]) -> str:
    """Convert None to empty string.

    Parameters
    ----------
    value: None or str
        Input value.

    Returns
    -------
    str
        Empty string.

    """
    if value is None:
        return EMPTY_STRING
    return value


def strip_whitespace(string: str) -> str:
    """Remove all whitespaces from string.

    Parameters
    ----------
    string: str
        Input string.

    Returns
    -------
    str
       Output string with whitespace removed.

    """
    return re.sub(re.compile(r"\s+"), "", string)


def is_non_empty_string(string: str) -> bool:
    """Return True if value == "", else False.

    Parameters
    ----------
    string: str
        Input string.

    Returns
    -------
    bool
        True if non-empty string, else False.

    """
    return not string == EMPTY_STRING


def normalize_special_characters(item: str) -> str:
    """Replace special characters with string equivalents.

    Parameters
    ----------
    item: str
        Input string.

    Returns
    -------
    str
        Output string after normalizing.

    """
    replacements = {
        "(": " ",
        ")": " ",
        ",": " ",
        "%": " percent ",
        "+": " plus ",
        "#": " number ",
        "&": " and ",
        "'s": "",
        "/": " per ",
    }
    for replacee, replacement in replacements.items():
        item = item.replace(replacee, replacement)

    item = item.strip()
    item = re.sub(r"\s+", "_", item)
    item = re.sub(r"[^0-9a-z_()]+", "_", item)
    item = re.sub(r"(?s:(^[0-9_].+))", "a_\1", item)
    return item


def count_occurrences(items: Iterable) -> List:
    """Count number of occurrences of the items.

    Parameters
    ----------
    items: Iterable
        Iterable of items to count the number of repeated values.

    Returns
    -------
    List
        (item, count) ordered by count, descending order.

    """
    counter = Counter(items)
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)


def convert_to_numeric(input_: Union[str, float, None]) -> Union[int, str, float]:
    """Convert input to numeric values.

    Parameters
    ----------
    input_: str
        Input value to try and convert to numeric.

    Returns
    -------
    str or int or float
        Converted numeric output.

    """
    if input_ in (None, np.nan):
        return np.nan
    if not isinstance(input_, str):
        # Originally this case implicitly returned None.
        raise TypeError(f"Expected string, received {type(input_)}")

    if is_range(input_):
        try:
            return compute_range_avg(input_)
        except Exception:
            print(input_)
            raise
    return re.sub("^-?[^0-9.]", "", str(input_))


def is_range(input_: str) -> bool:
    """Test if x matches range pattern.

    e.g. "2 to 5" or "2 - 5"

    Parameters
    ----------
    input_: str
        Input string to test if its a range.

    Returns
    -------
    bool
        True if categorical, False otherwise.

    """
    # [TODO:] Why is a space required? Isn't 1-5 all right, too?
    categorical_pattern = re.compile(r"-?\d+\s+(?:to|-)\s+(-?\d+)")
    return categorical_pattern.search(input_) is not None


def compute_range_avg(item: str) -> Union[int, float]:
    """Compute the average of a range.

    For instance, 5 - 7 -> 6, and 1 - 4 -> 2.5

    Parameters
    ----------
    item: str
        Input string which mentions a range.

    Returns
    -------
    int or float
        Computed average of range.

    """
    pattern_str = r"^(?P<first>-?\d+)\s*(?:to|-)\s*(?P<second>-?\d+)$"
    pattern = re.compile(pattern_str)
    if not (matched := pattern.search(item)):  # pylint: disable=superfluous-parens
        raise ValueError(f"'item' does not match expected pattern {pattern_str}")
    return (int(matched.group("first")) + int(matched.group("second"))) / 2
