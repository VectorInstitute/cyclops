"""Diagnosis codes processor module."""

import logging
import re
from typing import Dict, Optional

import pandas as pd

from cyclops.process.column_names import DIAGNOSIS_TRAJECTORY
from cyclops.process.constants import EMPTY_STRING, TRAJECTORIES
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def insert_decimal(input_: str, index: int = 2) -> str:
    """Insert decimal at index.

    Examples
    --------
    insert_decimal("232", 1) -> "2.32"

    Parameters
    ----------
    input_: str
        Input string.
    index: int, optional
        Index at which to insert decimal.

    Returns
    -------
    str:
        String after inserting decimal.

    """
    return input_[:index] + "." + input_[index:]


def get_alphabet(code: str) -> str:
    """Get alphabet occurring at the beginning of alphanumeric diagnosis code.

    Examples
    --------
    get_alphabet("M55") -> "M"

    Parameters
    ----------
    code: str
        Input diagnosis code.

    Returns
    -------
    str:
        Extracted alphabet.

    """
    return re.sub("[^a-zA-Z]", EMPTY_STRING, code).upper()


def get_numeric(code: str) -> str:
    """Get the numeric values from alphanumeric string which occur after an alphabet.

    Examples
    --------
    get_numeric("M55") -> "55"

    Parameters
    ----------
    code: str
        Input diagnosis code.

    Returns
    -------
    str:
        Extracted numeric.

    """
    return re.sub("[^0-9]", EMPTY_STRING, code)


def get_icd_category(code: str, trajectories: dict, raise_err: bool = False) -> str:
    """Get ICD10 category.

    Parameters
    ----------
    code: str
        Input diagnosis code.
    trajectories: dict
        Dictionary mapping of ICD10 trajectories.
    raise_err: Flag to raise error if code cannot be converted (for debugging.)

    Returns
    -------
    str:
        Mapped ICD10 category code.

    """
    if code is None:
        return EMPTY_STRING
    code = str(code)

    for _, (code_low, code_high) in trajectories.items():
        icd_category = "_".join([code_low, code_high])
        code_letter = get_alphabet(code)
        code_high_letter = get_alphabet(code_high)
        if code_letter < code_high_letter:
            return icd_category
        if (code_letter == code_high_letter) and (
            int(float(insert_decimal(get_numeric(code), index=2)))
            <= int(get_numeric(code_high))
        ):
            return icd_category

    if raise_err:
        raise Exception(f"Code cannot be converted: {code}")
    return EMPTY_STRING


def process_diagnoses(
    series: pd.Series, trajectories: Optional[Dict] = None
) -> pd.Series:
    """Process diagnoses data (codes) into trajectories.

    Parameters
    ----------
    series: pd.Series
        Diagnosis code data.
    trajectories: dict, optional
        Mapping from code to trajectory.

    Returns
    -------
    pd.Series
        Diagnoses trajectories.

    """
    if trajectories is None:
        trajectories = TRAJECTORIES

    series = series.apply(get_icd_category, args=(trajectories,))
    series = series.rename(DIAGNOSIS_TRAJECTORY)

    return series
