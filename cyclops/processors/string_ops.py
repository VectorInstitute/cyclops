"""String operations used in data extraction."""

from collections import Counter
from typing import Iterable, List, Union

import numpy as np
import re

from cyclops.processors.constants import EMPTY_STRING


def find_string_match(search_string: str, search_terms: str) -> bool:
    """Find string terms in search string to see if there are matches.

    Parameters
    ----------
    search_string: str
        The string to search for possible matches.
    search_terms: str
        String terms to search x1|x2...|xn.

    Returns
    -------
    bool
        True if any matches were found, else False.
    """
    search_string = search_string.lower()
    x = re.search(search_terms, search_string)
    return True if x else False


def fix_inequalities(search_string: str) -> str:
    """Match result value, remove inequality symbols (<, >, =).

    For e.g.
    10.2, >= 10, < 10, 11 -> 10.2, 10, 10, 11

    Parameters
    ----------
    search_string: str
        The string to search for possible matches.

    Returns
    -------
    str
        Result value string is matched to regex, symbols removed.
    """
    search_string = search_string.lower()
    matches = re.search(
        r"^\s*(<|>)?=?\s*(-?\s*[0-9]+(?P<floating>\.)?(?(floating)[0-9]+|))\s*$",
        search_string,
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


def is_non_empty_value(value: str) -> bool:
    """Return True if value == '', else False.

    Parameters
    ----------
    value: str
        Result value of lab test.

    Returns
    -------
    bool
        True if non-empty string, else False.

    """
    return False if value == EMPTY_STRING else True


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


def convert_to_numeric(x):
    """Convert different strings to numeric values.

    Parameters
    ----------
    x: str
        Input string.

    Returns
    -------
    Union[int, float]
        Converted numeric output.
    """
    if x in (None, np.nan):
        return np.nan
    if not isinstance(x, str):
        # Originally this case implicitly returned None.
        raise TypeError(f"Expected string, received {type(x)}")

    if is_range(x):
        try:
            return compute_range_avg(x)
        except Exception:
            print(x)
            raise
    return re.sub("^-?[^0-9.]", "", str(x))


def is_range(x: str) -> bool:
    """Test if x matches range pattern.

    e.g. "2 to 5" or "2 - 5"

    Parameters
    ----------
    x: str
        Input string to test if its a range.

    Returns
    -------
    bool
        True if categorical, False otherwise.
    """
    # [TODO:] Why is a space required? Isn't 1-5 all right, too?
    categorical_pattern = re.compile(r"-?\d+\s+(?:to|-)\s+(-?\d+)")
    return categorical_pattern.search(x) is not None


def compute_range_avg(item: str) -> Union[int, float]:
    """Compute the average of a range.

    For instance, 5 - 7 -> 6, and 1 - 4 -> 2.5

    Parameters
    ----------
    item: str
        Input string which mentions a range.

    Returns
    -------
    Union[int, float]
        Computed average of range.
    """
    pattern_str = r"^(?P<first>-?\d+)\s*(?:to|-)\s*(?P<second>-?\d+)$"
    pattern = re.compile(pattern_str)
    if not (matched := pattern.search(item)):
        raise ValueError(f"'item' does not match expected pattern {pattern_str}")
    return (int(matched.group("first")) + int(matched.group("second"))) / 2
