"""Labs processor module."""

import logging
import re

import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.constants import (
    POSITIVE_RESULT_TERMS,
    NEGATIVE_RESULT_TERMS,
)
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    ADMIT_TIMESTAMP,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
)
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


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


def is_non_empty_value(result_value: str) -> bool:
    """Return True if value == '', else False.

    Parameters
    ----------
    result_value: str
        Result value of lab test.

    Returns
    -------
    bool
        True if non-empty string, else False."""
    return False if result_value == "" else True


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

    def process(self):
        """Process raw lab data towards making them feature-ready.

        Raw data -> Filter by time window -> Group by test name -> Auto-featurization.

        """
        LOGGER.info("Processing raw lab data...")
        LOGGER.info(
            f"# labs: {len(self.data)}, # encounters: {self.data[ENCOUNTER_ID].nunique()}"
        )
        LOGGER.info("Filtering labs within aggregation window...")
        self.data = filter_labs_in_window(self.data)
        LOGGER.info(
            f"# labs: {len(self.data)}, # encounters: {self.data[ENCOUNTER_ID].nunique()}"
        )
        LOGGER.info("Removing labs with empty result values...")
        self.data = self.data[
            self.data[LAB_TEST_RESULT_VALUE].apply(is_non_empty_value)
        ].copy()
        LOGGER.info(
            f"# labs: {len(self.data)}, # encounters: {self.data[ENCOUNTER_ID].nunique()}"
        )
        self._featurize()

    def _featurize(self):
        """For each test, create appropriate features."""
        grouped_labs = self.data.groupby([LAB_TEST_NAME])


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
    labs_processor = LabsProcessor(data, must_have_columns)
    labs_processor.process()
