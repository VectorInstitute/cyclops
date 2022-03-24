"""Administrative data processor."""

import logging

import pandas as pd

from codebase_ops import get_log_file_path

from cyclops.processors.base import Processor
from cyclops.processors.column_names import ENCOUNTER_ID, AGE, SEX
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class AdminProcessor(Processor):
    """Admin data processor class."""

    @time_function
    def process(self) -> pd.DataFrame:
        """Process raw admin data to make them feature-ready.

        Returns
        -------
        pandas.DataFrame:
            Processed admin data.

        """
        self._log_counts_step("Processing raw admin data...")

        return self._create_features()

    def _create_features(self) -> pd.DataFrame:
        """Create admin features.

        Current support for ['age', 'sex']

        Returns
        -------
        pandas.DataFrame:
            Processed admin features.

        """
        encounters = list(self.data[ENCOUNTER_ID].unique())
        admin_col_names = [AGE, SEX]
        LOGGER.info(
            "# admin features: %d, # encounters: %d",
            len(admin_col_names),
            len(encounters),
        )
        features = pd.DataFrame(index=encounters, columns=admin_col_names)

        grouped = self.data.groupby([ENCOUNTER_ID])
        for encounter_id, admin in grouped:
            for admin_col_name in admin_col_names:
                assert admin[admin_col_name].nunique() == 1
                features.loc[encounter_id, admin_col_name] = admin[
                    admin_col_name
                ].unique()[0]

        return features
