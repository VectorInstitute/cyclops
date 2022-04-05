"""Vitals processor module."""

# mypy: ignore-errors

import logging

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.base import Processor
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_TIMESTAMP,
    VITAL_MEASUREMENT_VALUE,
)
from cyclops.processors.common import filter_within_admission_window
from cyclops.processors.constants import NEGATIVE_RESULT_TERMS, POSITIVE_RESULT_TERMS
from cyclops.processors.string_ops import fill_missing_with_nan, find_string_match
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class VitalsProcessor(Processor):
    """Vitals processor class."""

    @time_function
    def process(self) -> pd.DataFrame:
        """Process raw vitals data towards making them feature-ready.

        Returns
        -------
        pandas.DataFrame:
            Processed lab features.

        """
        self._log_counts_step("Processing raw vitals data...")

        self._aggregate_vitals()
        self._drop_unsupported_vitals()
        self._normalize_values()

        return self._create_features()

    def _normalize_values(self) -> None:
        """Normalize vital values, e.g. fill empty strings with NaNs."""
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

        self.data[VITAL_MEASUREMENT_VALUE] = self.data[VITAL_MEASUREMENT_VALUE].apply(
            fill_missing_with_nan
        )
        self._log_counts_step("Fill empty result string values with NaN...")

        self.data[VITAL_MEASUREMENT_VALUE] = self.data[VITAL_MEASUREMENT_VALUE].astype(
            "float"
        )
        LOGGER.info("Converting string result values to numeric...")

    def _drop_unsupported_vitals(self) -> None:
        """Drop some vitals currently not supported."""
        self.data = self.data[
            ~self.data[VITAL_MEASUREMENT_NAME].apply(
                find_string_match, args=("oxygen",)
            )
        ]
        self._log_counts_step(
            "Drop oxygen flow rate, saturation samples (unsupported)..."
        )

    def _aggregate_vitals(self) -> None:
        """Aggregate vitals based on static or time-interval windows."""
        self.data = filter_within_admission_window(
            self.data, VITAL_MEASUREMENT_TIMESTAMP
        )
        self._log_counts_step("Aggregating vitals within aggregation window...")

    def _create_features(self) -> pd.DataFrame:
        """For each vital measurement, create appropriate features.

        Returns
        -------
        pandas.DataFrame:
            Processed vitals.

        """
        LOGGER.info("Creating features...")
        vitals_names = list(self.data[VITAL_MEASUREMENT_NAME].unique())
        encounters = list(self.data[ENCOUNTER_ID].unique())
        LOGGER.info(
            "# vitals features: %d, # encounters: %d",
            len(vitals_names),
            len(encounters),
        )
        features = pd.DataFrame(index=encounters, columns=vitals_names)

        grouped_vitals = self.data.groupby([ENCOUNTER_ID, VITAL_MEASUREMENT_NAME])
        for (encounter_id, vital_name), vitals in grouped_vitals:
            features.loc[encounter_id, vital_name] = vitals[
                VITAL_MEASUREMENT_VALUE
            ].mean()

        return features
