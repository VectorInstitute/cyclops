"""Vitals processor module."""

import logging

import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_VALUE,
    VITAL_MEASUREMENT_TIMESTAMP,
)
from cyclops.processors.string_ops import is_non_empty_value, find_string_match
from cyclops.processors.common import filter_within_admission_window
from cyclops.processors.constants import POSITIVE_RESULT_TERMS, NEGATIVE_RESULT_TERMS
from cyclops.utils.log import setup_logging, LOG_FILE_PATH
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


class VitalsProcessor(Processor):
    """Vitals processor class."""

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
        """Process raw vitals data towards making them feature-ready.

        Returns
        -------
        pandas.DataFrame:
            Processed lab features.

        """
        self._log_counts_step("Processing raw vitals data...")
        self.data = filter_within_admission_window(
            self.data, VITAL_MEASUREMENT_TIMESTAMP  # type: ignore
        )
        self._log_counts_step("Filtering vitals within aggregation window...")

        self.data = self.data[
            ~self.data[VITAL_MEASUREMENT_NAME].apply(
                find_string_match, args=("oxygen",)
            )
        ].copy()

        # TODO: Add special processing to handle oxygen flow rate, saturation.
        self._log_counts_step("Drop oxygen flow rate, saturation samples...")

        self.data[VITAL_MEASUREMENT_VALUE][
            self.data[VITAL_MEASUREMENT_VALUE].apply(
                find_string_match, args=("|".join(POSITIVE_RESULT_TERMS),)
            )
        ] = "1"
        self.data[VITAL_MEASUREMENT_VALUE][
            self.data[VITAL_MEASUREMENT_VALUE].apply(
                find_string_match, args=("|".join(NEGATIVE_RESULT_TERMS),)
            )
        ] = "0"
        self._log_counts_step("Convert Positive/Negative to 1/0...")

        self.data[VITAL_MEASUREMENT_VALUE] = (
            self.data[VITAL_MEASUREMENT_VALUE].astype("float").copy()
        )
        LOGGER.info("Converting string result values to numeric...")

        self.data = self.data[
            self.data[VITAL_MEASUREMENT_VALUE].apply(is_non_empty_value)
        ].copy()
        self._log_counts_step("Removing vitals with empty result values...")

        LOGGER.info("Creating features...")
        return self._featurize()

    def _featurize(self) -> pd.DataFrame:
        """For each vital measurement, create appropriate features.

        Returns
        -------
        pandas.DataFrame:
            Processed vitals.

        """
        vitals_names = list(self.data[VITAL_MEASUREMENT_NAME].unique())
        encounters = list(self.data[ENCOUNTER_ID].unique())
        LOGGER.info(
            "# vitals features: %d, # encounters: %d",
            len(vitals_names),
            len(encounters)
        )
        features = pd.DataFrame(index=encounters, columns=vitals_names)

        grouped_vitals = self.data.groupby([ENCOUNTER_ID, VITAL_MEASUREMENT_NAME])
        for (encounter_id, vital_name), vitals in grouped_vitals:
            features.loc[encounter_id, vital_name] = vitals[
                VITAL_MEASUREMENT_VALUE
            ].mean()

        return features
