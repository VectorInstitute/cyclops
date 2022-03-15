"""Labs processor module."""

# mypy: ignore-errors

import logging

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.base import Processor
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
)
from cyclops.processors.string_ops import (
    fix_inequalities,
    to_lower,
    strip_whitespace,
    remove_text_in_parentheses,
    fill_missing_with_nan,
)
from cyclops.processors.common import filter_within_admission_window
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


UNSUPPORTED = [
    "urinalysis",
    "alp",
    "alt",
    "ast",
    "d-dimer",
    "ldh",
    "serum osmolality",
    "tsh",
    "urine osmolality",
]


def is_supported(lab_test_name: str) -> bool:
    """Check if lab test is supported (due to units not converted yet).

    Parameters
    ----------
    lab_test_name: str
        Name of lab test.

    Returns
    -------
    bool
        If supported return True, else False.
    """
    return lab_test_name not in UNSUPPORTED


class LabsProcessor(Processor):
    """Labs processor class."""

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

        self._aggregate_labs()
        self._normalize_lab_names()
        self._drop_unsupported_labs()
        self._normalize_result_values()
        self._normalize_units()

        return self._create_features()

    def _normalize_result_values(self) -> None:
        """Normalize result values, e.g. remove linequalities, convert to SI units."""
        self.data[LAB_TEST_RESULT_VALUE] = self.data[LAB_TEST_RESULT_VALUE].apply(
            fix_inequalities
        )
        self._log_counts_step("Fixing inequalities and removing outlier values...")

        self.data[LAB_TEST_RESULT_VALUE] = self.data[LAB_TEST_RESULT_VALUE].apply(
            fill_missing_with_nan
        )
        self._log_counts_step("Fill empty result string values with NaN...")

        LOGGER.info("Converting string result values to numeric...")
        self.data[LAB_TEST_RESULT_VALUE] = self.data[LAB_TEST_RESULT_VALUE].astype(
            "float"
        )

    def _normalize_units(self) -> None:
        """Normalize units strings, convert units to SI, result values accordingly."""
        LOGGER.info("Cleaning units and converting to SI...")
        self.data[LAB_TEST_RESULT_UNIT] = self.data[LAB_TEST_RESULT_UNIT].apply(
            to_lower
        )
        self.data[LAB_TEST_RESULT_UNIT] = self.data[LAB_TEST_RESULT_UNIT].apply(
            strip_whitespace
        )

    def _normalize_lab_names(self) -> None:
        """Normalize lab test names, e.g. remove parentheses make lower case."""
        self.data[LAB_TEST_NAME] = self.data[LAB_TEST_NAME].apply(
            remove_text_in_parentheses
        )
        self.data[LAB_TEST_NAME] = self.data[LAB_TEST_NAME].apply(to_lower)
        self._log_counts_step(
            "Remove text in parentheses and normalize lab test names..."
        )

    def _aggregate_labs(self) -> None:
        """Aggregate labs based on static or time-interval windows."""
        self.data = filter_within_admission_window(self.data, LAB_TEST_TIMESTAMP)
        self._log_counts_step("Aggregating labs within aggregation window...")

    def _drop_unsupported_labs(self) -> None:
        """Drop some labs currently not supported."""
        self.data = self.data.loc[self.data[LAB_TEST_NAME].apply(is_supported)]
        self._log_counts_step("Drop unsupported...")

    def _create_features(self) -> pd.DataFrame:
        """Create features, grouping by test and gathering result values.

        Returns
        -------
        pandas.DataFrame:
            Processed lab features.

        """
        LOGGER.info("Creating features...")
        lab_tests = list(self.data[LAB_TEST_NAME].unique())
        LOGGER.info(lab_tests)
        encounters = list(self.data[ENCOUNTER_ID].unique())
        LOGGER.info(
            "# labs features: %d, # encounters: %d", len(lab_tests), len(encounters)
        )
        features = pd.DataFrame(index=encounters, columns=lab_tests)

        grouped_labs = self.data.groupby([ENCOUNTER_ID, LAB_TEST_NAME])
        for (encounter_id, lab_test_name), labs in grouped_labs:
            features.loc[encounter_id, lab_test_name] = labs[
                LAB_TEST_RESULT_VALUE
            ].mean()

        return features
