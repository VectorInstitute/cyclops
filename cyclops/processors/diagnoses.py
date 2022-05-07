"""Diagnosis codes processor module."""

import logging
import re
from typing import Dict, Optional

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    DIAGNOSIS_CODE,
    DIAGNOSIS_TRAJECTORIES,
    ENCOUNTER_ID,
)
from cyclops.processors.constants import EMPTY_STRING, TRAJECTORIES
from cyclops.processors.util import log_counts_step
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def insert_decimal(input_: str, index: int = 2) -> str:
    """Insert decimal at index.

    Example
    -------
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

    Example
    -------
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

    Example
    -------
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


@time_function
def group_diagnosis_codes_to_trajectories(
    data: pd.DataFrame, trajectories: Optional[Dict] = None
) -> pd.DataFrame:
    """Process raw ICD diagnosis codes into grouped trajectories (one-hot encoded).

    For each encounter, a patient may have associated diagnoses information,
    and these codes can be grouped into ICD trajectories which allows them to
    be used as features or targets.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data with diagnoses codes associated to patient.

    Returns
    -------
    pandas.DataFrame:
        One-hot encoded binary ICD features.

    """
    log_counts_step(data, "Processing raw diagnosis codes...")
    if not trajectories:
        trajectories = TRAJECTORIES

    data[DIAGNOSIS_TRAJECTORIES] = data[DIAGNOSIS_CODE].apply(
        get_icd_category, args=(trajectories,)
    )
    log_counts_step(data, "Grouping ICD codes to trajectories...")

    encounters = list(data[ENCOUNTER_ID].unique())
    icd_trajectories = list(data[DIAGNOSIS_TRAJECTORIES].unique())
    LOGGER.info(
        "# diagnosis features: %d, # encounters: %d",
        len(icd_trajectories),
        len(encounters),
    )
    features = pd.crosstab(data[ENCOUNTER_ID], data[DIAGNOSIS_TRAJECTORIES])
    features.fillna(0, inplace=True)
    features = features.applymap(lambda x: int(x > 0))

    return features
