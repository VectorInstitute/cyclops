"""Base processor module."""

import logging

import pandas as pd

from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def _check_must_have_columns(data: pd.DataFrame, must_have_columns: list) -> None:
    """Check if data has required columns for processing.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    must_have_columns: list
        List of column names that must be present in data.
    """
    column_names = list(data.columns)
    for column_name in must_have_columns:
        assert column_name in column_names


def _gather_required_columns(
    data: pd.DataFrame, required_columns: list
) -> pd.DataFrame:
    """Gather only required columns and discard rest.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame to check.
    required_columns: list
        List of column names to keep in data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with required columns, other columns discarded.

    """
    return data[required_columns].copy()


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
        _check_must_have_columns(data, must_have_columns)

        self.data = _gather_required_columns(data, must_have_columns)
        self.must_have_columns = must_have_columns

    def print_data(self) -> None:
        """Print data."""
        LOGGER.info(self.data)

    def _log_counts_step(self, step_description: str) -> None:
        """Log num. of encounters and num. of samples (rows).

        Parameters
        ----------
        step_description: Description of intermediate processing step.

        """
        LOGGER.info(step_description)
        num_samples = len(self.data)
        num_encounters = self.data[ENCOUNTER_ID].nunique()
        LOGGER.info(f"# samples: {num_samples}, # encounters: {num_encounters}")
