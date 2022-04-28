"""Process raw data columns into outcomes of interest."""


import logging

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import (
    DISCHARGE_DISPOSITION,
    ENCOUNTER_ID,
    MORTALITY_IN_HOSPITAL,
)
from cyclops.processors.constants import MORTALITY_DISCHARGE_DISPOSITION_CODES
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def generate_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    """Process some raw data columns, into outcomes of interest.

    If column is present, it is processed into corresponding outcomes.
    For e.g. if discharge disposition codes are passed in the input data,
    'mortality in hospital' is obtained from the codes and added as an outcome
    column.

    Parameters
    ----------
    data: pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame or None
       Outcomes for the encounters.

    """
    encounters = list(data[ENCOUNTER_ID].unique())
    outcomes = pd.DataFrame(index=encounters)
    if DISCHARGE_DISPOSITION in data:
        mortality_in_hospital = data[DISCHARGE_DISPOSITION].apply(
            lambda code, codes: code in codes,
            args=(MORTALITY_DISCHARGE_DISPOSITION_CODES,),
        )
        mortality_in_hospital.columns = MORTALITY_IN_HOSPITAL
        outcomes = pd.concat([outcomes, mortality_in_hospital], axis=1)

    if len(outcomes.columns) > 0:
        return outcomes

    return None
