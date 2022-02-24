"""Diagnosis codes processor module."""

import logging
import re

import numpy as np
import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.feature import FeatureStore
from cyclops.processors.column_names import ENCOUNTER_ID, DIAGNOSIS_CODE
from cyclops.processors.constants import TRAJECTORIES, EMPTY_STRING
from cyclops.processors.utils import is_non_empty_value
from cyclops.utils.log import setup_logging, LOG_FILE_PATH
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def insert_decimal(input_: str, index: int = 2):
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


def get_code_letter(code):
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


def get_code_numerics(code):
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


def get_icd_category(
    code: str, trajectories: dict = TRAJECTORIES, raise_err: bool = False
):
    """Get ICD10 category.

    code: str
        Input diagnosis code.
    trajectories: dict, optional
        Dictionary mapping of ICD10 trajectories.
    raise_err: Flag to raise error if code cannot be converted (for debugging.)

    Returns
    -------
    str:
        Mapped ICD10 category code.

    """
    if code is None:
        return EMPTY_STRING

    try:
        code = str(code)
    except Exception:
        return EMPTY_STRING

    for item, (code_low, code_high) in trajectories.items():
        icd_category = "_".join([code_low, code_high])
        code_letter = get_code_letter(code)
        code_low_letter = get_code_letter(code_low)
        code_high_letter = get_code_letter(code_high)
        if code_letter > code_low_letter:
            pass
        elif (code_letter == code_low_letter) and (
            float(insert_decimal(get_code_numerics(code), index=2))
            >= int(get_code_numerics(code_low))
        ):
            pass
        else:
            continue
        if code_letter < code_high_letter:
            return icd_category
        elif (code_letter == code_high_letter) and (
            int(float(insert_decimal(get_code_numerics(code), index=2)))
            <= int(get_code_numerics(code_high))
        ):
            return icd_category
        else:
            continue

    if raise_err:
        raise Exception("Code cannot be converted: {}".format(code))
    else:
        return EMPTY_STRING


class DiagnosisProcessor(Processor):
    """Diagnosis codes processor class."""

    def __init__(self, data: pd.DataFrame, must_have_columns: list):
        """Instantiate.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with raw features.
        must_have_columns: list
            List of column names of features that must be present in data.
        """
        super().__init__(data, must_have_columns)

    def _log_counts_step(self, step_description: str):
        """Log num. of encounters and num. of lab tests.

        Parameters
        ----------
        step_description: Description of intermediate processing step.

        """
        LOGGER.info(step_description)
        num_codes = len(self.data)
        num_encounters = self.data[ENCOUNTER_ID].nunique()
        LOGGER.info(f"# diagnosis codes: {num_codes}, # encounters: {num_encounters}")

    @time_function
    def process(self) -> np.ndarray:
        """Process raw diagnosis codes into ICD codes (one-hot encoded).

        Returns
        -------
        numpy.ndarray:
            One-hot encoded binary ICD features.

        """
        self._log_counts_step("Processing raw diagnosis codes...")
        self.data[DIAGNOSIS_CODE] = (
            self.data[DIAGNOSIS_CODE].apply(get_icd_category).copy()
        )

        self._log_counts_step("Converting diagnosis codes to ICD codes...")
        self.data = self.data[
            self.data[DIAGNOSIS_CODE].apply(is_non_empty_value)
        ].copy()
        self._log_counts_step("Removing unmapped, i.e. nan codes...")

        encounters = list(self.data[ENCOUNTER_ID].unique())
        icd_codes = list(self.data[DIAGNOSIS_CODE].unique())

        features = np.zeros((len(encounters), len(icd_codes)))
        features = pd.DataFrame(features, index=encounters, columns=icd_codes)

        grouped_codes = self.data.groupby([ENCOUNTER_ID])
        for encounter_id, codes in grouped_codes:
            icd_codes_encounter = codes[DIAGNOSIS_CODE]
            for icd_code_encounter in icd_codes_encounter:
                features.loc[encounter_id, icd_code_encounter] = 1

        return features


if __name__ == "__main__":
    data = pd.read_hdf(
        "/mnt/nfs/project/delirium/_extract/extract.h5",
        key="query_gemini_delirium_diagnosis",
    )
    must_have_columns = [ENCOUNTER_ID, DIAGNOSIS_CODE]
    feature_store = FeatureStore()
    diagnosis_processor = DiagnosisProcessor(data, must_have_columns)
    diagnosis_features = diagnosis_processor.process()
    feature_store.add_features(diagnosis_features)
    print(feature_store.df)
