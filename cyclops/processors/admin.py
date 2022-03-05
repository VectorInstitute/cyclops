"""Administrative data processor."""

import logging

import pandas as pd

from cyclops.processors.base import Processor
from cyclops.processors.column_names import ENCOUNTER_ID, AGE, SEX
from cyclops.processors.constants import TRAJECTORIES, EMPTY_STRING
from cyclops.processors.string_ops import is_non_empty_value

from cyclops.utils.log import setup_logging, LOG_FILE_PATH
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


class AdminProcessor(Processor):
    """Admin data processor class."""

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

    @time_function
    def process(self) -> pd.DataFrame:
        """Process raw admin data to make them feature-ready.

        Returns
        -------
        pandas.DataFrame:
            Processed admin data.

        """
        self._log_counts_step("Processing raw admin data...")

        encounters = list(self.data[ENCOUNTER_ID].unique())
        admin_column_names = [AGE, SEX]
        LOGGER.info(
            f"# admin features: {len(admin_column_names)}, # encounters: {len(encounters)}"
        )
        features = pd.DataFrame(index=encounters, columns=admin_column_names)

        grouped_admin = self.data.groupby([ENCOUNTER_ID])
        for encounter_id, admin in grouped_admin:
            for admin_column_name in admin_column_names:
                assert admin[admin_column_name].nunique() == 1
                features.loc[encounter_id, admin_column_name] = admin[
                    admin_column_name
                ].unique()[0]

        return features
