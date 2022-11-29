"""Post-processing functions applied to queried MIMIC data (Pandas DataFrames)."""

import pandas as pd

from cyclops.process.column_names import CARE_UNIT
from cyclops.process.constants import ER, ICU, IP, SCU
from cyclops.query.postprocess.util import process_care_unit_changepoints
from cyclops.utils.profile import time_function

CARE_UNIT_MAP = {
    IP: {
        "observation": ["Observation", "Psychiatry"],
        "medicine": ["Medicine", "Medical/Surgical (Gynecology)"],
    },
    ER: {
        "er": ["Emergency Department", "Emergency Department Observation"],
    },
    ICU: {
        "icu": [
            "Surgical Intensive Care Unit (SICU)",
            "Medical/Surgical Intensive Care Unit (MICU/SICU)",
            "Medical Intensive Care Unit (MICU)",
            "Trauma SICU (TSICU)",
            "Neuro Surgical Intensive Care Unit (Neuro SICU)",
            "Cardiac Vascular Intensive Care Unit (CVICU)",
        ],
    },
    SCU: {
        "surgery": [
            "Med/Surg",
            "Surgery",
            "Surgery/Trauma",
            "Med/Surg/Trauma",
            "Med/Surg/GYN",
            "Surgery/Vascular/Intermediate",
            "Thoracic Surgery",
            "Transplant",
            "Cardiac Surgery",
            "PACU",
            "Surgery/Pancreatic/Biliary/Bariatric",
        ],
        "cardiology": [
            "Cardiology",
            "Coronary Care Unit (CCU)",
            "Cardiology Surgery Intermediate",
            "Medicine/Cardiology",
            "Medicine/Cardiology Intermediate",
        ],
        "vascular": [
            "Vascular",
            "Hematology/Oncology",
            "Hematology/Oncology Intermediate",
        ],
        "neuro": ["Neurology", "Neuro Intermediate", "Neuro Stepdown"],
        "neonatal": [
            "Obstetrics (Postpartum & Antepartum)",
            "Neonatal Intensive Care Unit (NICU)",
            "Special Care Nursery (SCN)",
            "Nursery - Well Babies",
            "Obstetrics Antepartum",
            "Obstetrics Postpartum",
            "Labor & Delivery",
        ],
    },
}
NONSPECIFIC_CARE_UNIT_MAP = {
    "medicine": IP,
    "observation": IP,
    "er": ER,
    "icu": ICU,
    "cardiology": SCU,
    "neuro": SCU,
    "neonatal": SCU,
    "surgery": SCU,
    "vascular": SCU,
}
CARE_UNIT_HIERARCHY = [ER, ICU, SCU, IP]


def process_mimic_care_unit_changepoints(data: pd.DataFrame) -> pd.DataFrame:
    """Process MIMIC changepoint care unit information in a hierarchical fashion.

    Using the admit, discharge, and care unit information, create a
    changepoint DataFrame usable for aggregation labelling purposes.
    If a patient is in multiple care units at a changepoint, the care
    unit highest in the hierarchy is selected.

    Parameters
    ----------
    data: pandas.DataFrame
        The admit, discharge, and care unit information for a single encounter.
        Expects columns "admit", "discharge", and CARE_UNIT.

    Returns
    -------
    pandas.DataFrame
        Changepoint information with associated care unit.

    """
    return process_care_unit_changepoints(data, CARE_UNIT_HIERARCHY)


@time_function
def process_mimic_care_units(
    transfers: pd.DataFrame, specific: bool = False
) -> pd.DataFrame:
    """Process care unit data.

    Processes the MIMIC Transfers table into a cleaned and simplified care
    units DataFrame.

    Parameters
    ----------
    transfers : pandas.DataFrame
        MIMIC transfers table as a DataFrame.
    specific : bool, optional
        Whether care_unit_name column has specific or non-specific care units.

    Returns
    -------
    pandas.DataFrame
        Processed care units for MIMIC encounters.

    """
    transfers.rename(
        columns={
            "intime": "admit",
            "outtime": "discharge",
            "careunit": CARE_UNIT,
        },
        inplace=True,
    )

    # Drop rows with eventtype discharge.
    # Its admit timestamp is the discharge timestamp of eventtype admit.
    transfers = transfers[transfers["eventtype"] != "discharge"]
    transfers = transfers.drop("eventtype", axis=1)
    transfers = transfers[transfers[CARE_UNIT] != "Unknown"]

    # Create replacement dictionary for care unit categories depending on specificity.
    replace_dict = {}
    for unit, unit_dict in CARE_UNIT_MAP.items():
        for specific_unit, unit_list in unit_dict.items():
            value = specific_unit if specific else unit
            replace_dict.update({elem: value for elem in unit_list})
    transfers[CARE_UNIT].replace(replace_dict, inplace=True)

    transfers.dropna(inplace=True)

    return transfers
