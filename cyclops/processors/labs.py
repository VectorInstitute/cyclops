"""Labs processor module."""

import logging

import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.feature_handler import FeatureHandler
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    ADMIT_TIMESTAMP,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
    REFERENCE_RANGE,
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
        Remove empty values, strings -> convert units -> Feature handler

        """
        self._log_counts_step("Processing raw lab data...")
        self.data = filter_within_admission_window(self.data, LAB_TEST_TIMESTAMP)
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

    def _featurize(self):
        """For each test, create appropriate features."""
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
        REFERENCE_RANGE,
    ]
    feature_handler = FeatureHandler()
    labs_processor = LabsProcessor(data, must_have_columns)
    lab_features = labs_processor.process()
    feature_handler.add_features(lab_features)
    print(feature_handler.df)
