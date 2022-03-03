"""Labs processor module."""

import logging

import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
)
from cyclops.processors.string_ops import (
    is_non_empty_value,
    fix_inequalities,
    to_lower,
    strip_whitespace,
)
from cyclops.processors.common import filter_within_admission_window
from cyclops.utils.log import setup_logging, LOG_FILE_PATH
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


class LabsProcessor(Processor):
    """Labs processor class."""

    def __init__(self, data: pd.DataFrame, must_have_columns: list) -> None:
        """Instantiate.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with raw features.
        must_have_columns: list
            List of column names of features that must be present in data.
        """
        super().__init__(data, must_have_columns)

    @time_function
    def process(self) -> pd.DataFrame:
        """Process raw lab data towards making them feature-ready.

        Raw data -> Filter by time window -> Remove inequalities ->
        Remove empty values, strings -> Convert units -> Feature handler

        Returns
        -------
        pandas.DataFrame:
            Processed lab features.

        """
        self._log_counts_step("Processing raw lab data...")
        self.data = filter_within_admission_window(
            self.data, LAB_TEST_TIMESTAMP  # type: ignore
        )
        self._log_counts_step("Filtering labs within aggregation window...")
        self.data[LAB_TEST_RESULT_VALUE] = (
            self.data[LAB_TEST_RESULT_VALUE].apply(fix_inequalities).copy()
        )
        # TODO: Handle some of the string matches.
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

    def _featurize(self) -> pd.DataFrame:
        """For each test, create appropriate features.

        Returns
        -------
        pandas.DataFrame:
            Processed lab features.

        """
        lab_tests = list(self.data[LAB_TEST_NAME].unique())
        encounters = list(self.data[ENCOUNTER_ID].unique())
        LOGGER.info(
            f"# labs features: {len(lab_tests)}, # encounters: {len(encounters)}"
        )
        features = pd.DataFrame(index=encounters, columns=lab_tests)

        grouped_labs = self.data.groupby([ENCOUNTER_ID, LAB_TEST_NAME])
        for (encounter_id, lab_test_name), labs in grouped_labs:
            features.loc[encounter_id, lab_test_name] = labs[
                LAB_TEST_RESULT_VALUE
            ].mean()

        return features
