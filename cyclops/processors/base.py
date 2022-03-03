"""Base processor module."""

import logging

import pandas as pd

from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


class Processor:
    """Base processor.

    Attributes
    ----------
    data: pandas.DataFrame
        Dataframe with raw features.
    must_have_columns: list
        List of column names that must be present in data.
    """

    def __init__(self, data: pd.DataFrame, must_have_columns: list) -> None:
        """Instantiate.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with raw features.
        must_have_columns: list
            List of column names of features that must be present in data.
        """
        assert type(data) is pd.DataFrame
        assert type(must_have_columns) is list
        self.data = data.copy()
        self.must_have_columns = must_have_columns

        self._check_must_have_columns()
        self._gather_required_columns()

    def _log_counts_step(self, step_description: str) -> None:
        """Log num. of encounters and num. of lab tests.

        Parameters
        ----------
        step_description: Description of intermediate processing step.

        """
        LOGGER.info(step_description)
        num_samples = len(self.data)
        num_encounters = self.data[ENCOUNTER_ID].nunique()
        LOGGER.info(f"# samples: {num_samples}, # encounters: {num_encounters}")

    def _check_must_have_columns(self) -> None:
        """Check if data has required columns for processing."""
        column_names = list(self.data.columns)
        for column_name in self.must_have_columns:
            assert column_name in column_names

    def _gather_required_columns(self) -> None:
        """Gather only required columns and discard rest."""
        self.data = self.data[self.must_have_columns].copy()
