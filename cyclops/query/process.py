"""Post-processing functions applied to queried data (pandas dataframes)."""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.pyplot import figure
from pandas import Timestamp

from cyclops.processors.column_names import CARE_UNIT
from cyclops.processors.constants import ER, ICU, IP, SCU
from cyclops.processors.util import check_must_have_columns
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
CARE_UNIT_HIERARCHY = {ER: 0, ICU: 1, SCU: 2, IP: 3}
CARE_UNIT_HIERARCHY_INV = {v: k for k, v in CARE_UNIT_HIERARCHY.items()}


def to_timestamp(data: Union[pd.Series, np.ndarray]) -> pd.Series:
    """Convert a Pandas series or NumPy array to a datetime/timestamp type.

    Parameters
    ----------
    data: pandas.Series or numpy.ndarray
        Data to be converted.

    Returns
    -------
    pandas.Series
        The converted data.

    """
    if isinstance(data, pd.Series):
        return pd.to_datetime(data)

    if isinstance(data, np.ndarray):
        return pd.to_datetime(pd.Series(data))

    raise ValueError(f"Type of data argument ({type(data)}) not supported.")


def event_time_between(
    event: Timestamp, start: pd.Series, end: pd.Series, inclusive: bool = True
) -> pd.Series:
    """Return whether an event time is between some start and end time.

    May also specify whether the comparison operators are inclusive or not..

    Parameters
    ----------
    event: pandas._libs.tslibs.timestamps.Timestamp
        Event time.
    start: pandas.Series
        A series of timestamps.
    end: pandas.Series
        A series of timestamps.

    Returns
    -------
    pandas.Series
        A boolean Series representing whether the event is between
        the start and end timestamps.

    """
    if inclusive:
        return (event >= start) & (event <= end)
    return (event > start) & (event < end)


def plot_admit_discharge(data, figsize=(10, 4)) -> None:
    """Plot a series of admit discharge times given a description.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame with 'admit', 'discharge', and 'description' columns.
        The admit and discharge columns must be convertable to Timestamps.

    """
    check_must_have_columns(
        data, ["admit", "discharge", "description"], raise_error=True
    )

    figure(figsize=figsize, dpi=80)
    colors = get_cmap("Accent").colors

    data["admit_int"] = to_timestamp(data["admit"]).astype(int)
    data["discharge_int"] = to_timestamp(data["discharge"]).astype(int)

    desc_dict = {}
    for val, key in enumerate(data["description"].unique()):
        desc_dict[key] = val

    data["admit_int"] = data["admit"].astype(int)
    data["discharge_int"] = data["discharge"].astype(int)

    plotted = []

    def plot_timerange(admit, discharge, description):
        ind = desc_dict[description]
        if description in plotted:
            plt.plot([admit, discharge], [ind, ind], color=colors[ind])
        else:
            plt.plot(
                [admit, discharge], [ind, ind], color=colors[ind], label=description
            )
            plotted.append(description)

    for _, row in data.iterrows():
        plot_timerange(row["admit_int"], row["discharge_int"], row["description"])
    plt.legend()


def process_changepoints(data):
    """Process changepoint care unit information in a hierarchical fashion.

    Using the admit, discharge, and care unit information, create a
    changepoint DataFrame usable for aggregation labelling purposes.

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
    changepoints = pd.concat([data["admit"], data["discharge"]])
    changepoints.sort_values(inplace=True)
    changepoints = changepoints.unique()

    data = []
    for changepoint in changepoints:
        is_between = event_time_between(changepoint, data["admit"], data["discharge"])
        careunits = data[is_between][CARE_UNIT].unique()
        careunit_nums = list(map(lambda x: CARE_UNIT_HIERARCHY[x], careunits))
        careunit_selected = CARE_UNIT_HIERARCHY_INV[min(careunit_nums)]
        data.append([changepoint, careunit_selected])

    return pd.DataFrame(data, columns={"changepoint", "care_unit"})


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
