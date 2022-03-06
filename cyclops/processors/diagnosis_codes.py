"""Diagnosis codes processor module."""

import logging
import re

import pandas as pd

from codebase_ops import get_log_file_path

from cyclops.processors.base import Processor
from cyclops.processors.column_names import ENCOUNTER_ID, DIAGNOSIS_CODE
from cyclops.processors.constants import TRAJECTORIES, EMPTY_STRING
from cyclops.processors.string_ops import is_non_empty_value
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def insert_decimal(input_: str, index: int = 2) -> str:
    """Insert decimal at index.

    Parameters
    ----------
    input_: str
        Input string.
    index: int
        Index at which to insert decimal.

    Returns
    -------
    str:
        String after inserting decimal.
    """
    return input_[:index] + "." + input_[index:]


def get_code_letter(code: str) -> str:
    """Get the letter from diagnosis code.

    E.g. M55 -> M

    Parameters
    ----------
    code: str
        Input diagnosis code.

    Returns
    -------
    str:
        Extracted letter.
    """
    return re.sub("[^a-zA-Z]", EMPTY_STRING, code).upper()


def get_code_numerics(code: str) -> str:
    """Get the numeric values from diagnosis code.

    E.g. M55 -> 55

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
        code_letter = get_code_letter(code)
        code_high_letter = get_code_letter(code_high)
        if code_letter < code_high_letter:
            return icd_category
        if (code_letter == code_high_letter) and (
            int(float(insert_decimal(get_code_numerics(code), index=2)))
            <= int(get_code_numerics(code_high))
        ):
            return icd_category

    if raise_err:
        raise Exception(f"Code cannot be converted: {code}")
    return EMPTY_STRING


class DiagnosisProcessor(Processor):
    """Diagnosis codes processor class."""

    @time_function
    def process(self) -> pd.DataFrame:
        """Process raw diagnosis codes into ICD codes (one-hot encoded).

        Returns
        -------
        pandas.DataFrame:
            One-hot encoded binary ICD features.

        """
        self._log_counts_step("Processing raw diagnosis codes...")
        self.data[DIAGNOSIS_CODE] = (  # type: ignore
            self.data[DIAGNOSIS_CODE]  # type: ignore
            .apply(get_icd_category, args=(TRAJECTORIES,))
            .copy()
        )

        self._log_counts_step("Converting diagnosis codes to ICD codes...")
        self.data = self.data[  # type: ignore
            self.data[DIAGNOSIS_CODE].apply(is_non_empty_value)  # type: ignore
        ].copy()
        self._log_counts_step("Removing unmapped, i.e. nan codes...")

        encounters = list(self.data[ENCOUNTER_ID].unique())
        icd_codes = list(self.data[DIAGNOSIS_CODE].unique())

        features = pd.DataFrame(index=encounters, columns=icd_codes)

        grouped_codes = self.data.groupby([ENCOUNTER_ID])
        for encounter_id, codes in grouped_codes:
            icd_codes_encounter = codes[DIAGNOSIS_CODE]
            for icd_code_encounter in icd_codes_encounter:
                features.loc[encounter_id, icd_code_encounter] = 1

        return features
