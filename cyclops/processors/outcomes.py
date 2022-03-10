"""Outcomes of interest processor."""

import logging

import pandas as pd

from codebase_ops import get_log_file_path

from cyclops.processors.base import Processor
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    DISCHARGE_DISPOSITION,
    LENGTH_OF_STAY_IN_ER,
    MORTALITY_IN_HOSPITAL
)
from cyclops.processors.constants import MORTALITY_DISCHARGE_DISPOSITION
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def is_code_in_list(code: float, codes:list) -> bool:
    """Check if a given numeric value is in a list of numeric codes.
    
    Parameters
    ----------
    code: float
        Input code to check.
    codes: list
        List of codes in which we would like to see if input code exists.
    
    Returns
    -------
    bool
        True if code is in list, else False.
    """
    return code in codes
    

class OutcomesProcessor(Processor):
    """Outcomes processor class."""

    @time_function
    def process(self) -> pd.DataFrame:
        """Process raw outcomes information to make them feature/target-ready.

        Returns
        -------
        pandas.DataFrame:
            Processed outcomes data.

        """
        self._log_counts_step("Processing raw outcomes data...")

        return self._create_features()
    
    def _extract_mortality_in_hospital(self):
        """Check if discharge disposition codes for mortality."""
        discharge_disposition = self.data[DISCHARGE_DISPOSITION].copy()
        is_mortality = discharge_disposition.apply(is_code_in_list, args=([7],))
        is_mortality = is_mortality.rename(MORTALITY_IN_HOSPITAL)
        return is_mortality
                                                   
    def _create_features(self) -> pd.DataFrame:
        """Create outcomes features (targets).

        Current support for:
        1. Mortality in hospital obtained from discharge disposition code.
        2. LOS (ER) duration_er_stay_derived

        Returns
        -------
        pandas.DataFrame:
            Processed outcomes features.
        """ 
        encounters = list(self.data[ENCOUNTER_ID].unique())
        outcomes_col_names = []
        features = pd.DataFrame(index=encounters)

        if DISCHARGE_DISPOSITION in self.must_have_columns:
            is_mortality = self._extract_mortality_in_hospital()
            outcomes_col_names.append(MORTALITY_IN_HOSPITAL)
            is_mortality.index = encounters
            features = pd.concat([features, is_mortality], axis=1)
        if LENGTH_OF_STAY_IN_ER in self.must_have_columns:
            outcomes_col_names.append(LENGTH_OF_STAY_IN_ER)
            los_er = self.data[LENGTH_OF_STAY_IN_ER].copy()
            los_er.index = encounters
            features = pd.concat([features, los_er], axis=1)
        
        return features
