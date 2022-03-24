"""Base processor module."""

import logging

import pandas as pd

from codebase_ops import get_log_file_path

from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


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


class Processor:  # pylint: disable=too-few-public-methods
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
        assert isinstance(data, pd.DataFrame)
        assert isinstance(must_have_columns, list)
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
        LOGGER.info("# samples: %d, # encounters: %d", num_samples, num_encounters)
