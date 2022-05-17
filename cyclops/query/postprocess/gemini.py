"""Post-processing functions applied to queried GEMINI data (Pandas DataFrames)."""

import pandas as pd

from cyclops.query.postprocess.util import process_care_unit_changepoints

CARE_UNIT_HIERARCHY = [
    "ER",
    "Emergency",
    "ICU",
    "SCU",
    "Peri-op",
    "Palliative",
    "Step-down",
    "Rehab",
    "Other ward",
    "GIM ward",
    "IP",
]


def process_gemini_care_unit_changepoints(data: pd.DataFrame) -> pd.DataFrame:
    """Process GEMINI changepoint care unit information in a hierarchical fashion.

    Using the admit, discharge, and care unit information, create a
    changepoint DataFrame usable for aggregation labelling purposes.
    If a patient is in multiple care units at a changepoint, the care
    unit highest in the hierarchy is selected.

    Parameters
    ----------
    data: pandas.DataFrame
        The admit, discharge, and care unit information for a single encounter.
        Expects columns "admit", "discharge", and CARE_UNIT.
    care_unit_hierarchy: list
        Ordered list of care units from most relevant to to least.

    Returns
    -------
    pandas.DataFrame
        Changepoint information with associated care unit.

    """
    return process_care_unit_changepoints(data, CARE_UNIT_HIERARCHY)
