"""Labs processor module."""

import logging
import re

import numpy as np
import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.feature import FeatureStore
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    ADMIT_TIMESTAMP,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
)
from cyclops.processors.utils import is_non_empty_value
from cyclops.utils.log import setup_logging, LOG_FILE_PATH
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def filter_labs_in_window(
    labs_data: pd.DataFrame, aggregation_window: int = 24
) -> pd.DataFrame:
    """Filter labs data based on single window value.

    For e.g. if window is 24 hrs, then all labs 24 hrs
    before time of admission and after 24 hrs of admission are considered.

    Parameters
    ----------
    labs_data: pandas.DataFrame
        Labs data before filtering.
    aggregation_window: int, optional
        Window (no. of hrs) before and after admission to consider.

    Returns
    -------
    pandas.DataFrame
        Filtered data frame, aggregated tests within window.
    """
    labs_df_filtered = labs_data.copy()
    sample_time = labs_df_filtered[LAB_TEST_TIMESTAMP]
    admit_time = labs_df_filtered[ADMIT_TIMESTAMP]
    window_condition = abs((sample_time - admit_time) / pd.Timedelta(hours=1))
    labs_df_filtered = labs_df_filtered.loc[window_condition <= aggregation_window]
    return labs_df_filtered


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


class LabsProcessor(Processor):
    """Labs processor class."""

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
        num_labs = len(self.data)
        num_encounters = self.data[ENCOUNTER_ID].nunique()
        LOGGER.info(f"# labs: {num_labs}, # encounters: {num_encounters}")

    @time_function
    def process(self):
        """Process raw lab data towards making them feature-ready.

        Raw data -> Filter by time window -> Remove inequalities ->
        Remove empty values, strings, convert units to lowercase -> Auto-featurization.

        """
        self._log_counts_step("Processing raw lab data...")
        self.data = filter_labs_in_window(self.data)
        self._log_counts_step("Filtering labs within aggregation window...")
        self.data[LAB_TEST_RESULT_VALUE] = (
            self.data[LAB_TEST_RESULT_VALUE].apply(fix_inequalities).copy()
        )
        self._log_counts_step("Fixing inequalities and removing outlier values...")
        self.data = self.data[
            self.data[LAB_TEST_RESULT_VALUE].apply(is_non_empty_value)
        ].copy()
        self._log_counts_step("Removing labs with empty result values...")

        LOGGER.info("Converting string result values to numeric...")
        self.data[LAB_TEST_RESULT_VALUE] = self.data[LAB_TEST_RESULT_VALUE].astype(
            "float"
        )

        LOGGER.info("Cleaning units and converting to SI...")
        self.data[LAB_TEST_RESULT_UNIT] = self.data[LAB_TEST_RESULT_UNIT].apply(
            to_lower
        )
        self.data[LAB_TEST_RESULT_UNIT] = self.data[LAB_TEST_RESULT_UNIT].apply(
            strip_whitespace
        )

        LOGGER.info("Creating features...")
        return self._featurize()

    def _featurize(self):
        """For each test, create appropriate features."""
        lab_tests = list(self.data[LAB_TEST_NAME].unique())
        encounters = list(self.data[ENCOUNTER_ID].unique())
        LOGGER.info(
            f"# labs features: {len(lab_tests)}, # encounters: {len(encounters)}"
        )
        features = np.zeros((len(encounters), len(lab_tests)))
        features = pd.DataFrame(features, index=encounters, columns=lab_tests)

        grouped_labs = self.data.groupby([ENCOUNTER_ID, LAB_TEST_NAME])
        for (encounter_id, lab_test_name), labs in grouped_labs:
            features.loc[encounter_id, lab_test_name] = labs[
                LAB_TEST_RESULT_VALUE
            ].mean()

        return features


if __name__ == "__main__":
    data = pd.read_hdf(
        "/mnt/nfs/project/delirium/_extract/extract.h5", key="query_gemini_delirium_lab"
    )
    must_have_columns = [
        ENCOUNTER_ID,
        ADMIT_TIMESTAMP,
        LAB_TEST_NAME,
        LAB_TEST_TIMESTAMP,
        LAB_TEST_RESULT_VALUE,
        LAB_TEST_RESULT_UNIT,
    ]
    feature_store = FeatureStore()
    labs_processor = LabsProcessor(data, must_have_columns)
    lab_features = labs_processor.process()
    feature_store.add_features(lab_features)
    print(feature_store.df)
