"""Post-processing functions applied to queried data (Pandas DataFrames)."""

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
    event: Timestamp,
    admit: pd.Series,
    discharge: pd.Series,
    admit_inclusive: bool = True,
    discharge_inclusive: bool = False
) -> pd.Series:
    """Return whether an event time is between some start and end time.

    May also specify whether the comparison operators are inclusive or not..

    Parameters
    ----------
    event: pandas._libs.tslibs.timestamps.Timestamp
        Event time.
    admit: pandas.Series
        A series of timestamps.
    discharge: pandas.Series
        A series of timestamps.
    admit_inclusive: bool
        Whether to have an inclusive inequality for the admit condition.
    discharge_inclusive: bool
        Whether to have an inclusive inequality for the discharge condition.
    
    Returns
    -------
    pandas.Series
        A boolean Series representing whether the event is between
        the start and end timestamps.

    """
    if admit_inclusive:
        admit_cond = event >= admit
    else:
        admit_cond = event > admit
    
    if discharge_inclusive:
        discharge_cond = event <= discharge
    else:
        discharge_cond = event < discharge
    
    return admit_cond & discharge_cond


def plot_admit_discharge(
    data: pd.DataFrame,
    description: str = "description",
    figsize: tuple = (10, 4)
) -> None:
    """Plot a series of admit discharge times given a description.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame with 'admit', 'discharge', and description columns.
        The admit and discharge columns must be convertable to Timestamps.

    """
    check_must_have_columns(
        data, ["admit", "discharge", description], raise_error=True
    )

    figure(figsize=figsize, dpi=80)
    colors = get_cmap("Accent").colors

    data["admit_int"] = to_timestamp(data["admit"]).astype(int)
    data["discharge_int"] = to_timestamp(data["discharge"]).astype(int)

    desc_dict = {}
    for val, key in enumerate(data[description].unique()):
        desc_dict[key] = val

    data["admit_int"] = data["admit"].astype(int)
    data["discharge_int"] = data["discharge"].astype(int)

    plotted = []

    def plot_timerange(admit, discharge, desc):
        ind = desc_dict[desc]
        if desc in plotted:
            plt.plot([admit, discharge], [ind, ind], color=colors[ind])
        else:
            plt.plot(
                [admit, discharge], [ind, ind], color=colors[ind], label=desc
            )
            plotted.append(desc)

    for _, row in data.iterrows():
        plot_timerange(row["admit_int"], row["discharge_int"], row[description])
    plt.legend()


def process_care_unit_changepoints(
    data: pd.DataFrame, care_unit_hierarchy: list
) -> pd.DataFrame:
    """Process changepoint care unit information in a hierarchical fashion.

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
        Changepoint information with associated care unit. The care unit
        information is relevant up until the next change point

    """
    # Define mapping dictionaries
    hierarchy = {care_unit_hierarchy[i]: i for i in range(len(care_unit_hierarchy))}
    hierarchy_inv = {i: care_unit_hierarchy[i] for i in range(len(care_unit_hierarchy))}
    
    # Create changepoints
    changepoints = pd.concat([data["admit"], data["discharge"]])
    changepoints.sort_values(inplace=True)
    changepoints = changepoints.unique()

    # Remove the final changepoint, which is the final discharge (has no careunit)
    changepoints = changepoints[:-1]
    
    # FIX: Discharge is counting as a changepoint which will be included in the is_between
    # when it shouldn't... For admit it should be ">= admit", but for discharge it should be
    # "< discharge"
    
    # Select the most relevant care unit for each changepoint
    changepoint_data = []
    for changepoint in changepoints:
        is_between = event_time_between(
            changepoint,
            data["admit"],
            data["discharge"],
            admit_inclusive=True,
            discharge_inclusive=False
        )
        care_units = data[is_between][CARE_UNIT].unique()
        if len(care_units) > 0:
            care_unit_inds = list(map(lambda x: hierarchy[x], care_units))
            care_unit_selected = hierarchy_inv[min(care_unit_inds)]
        else:
            care_unit_selected = np.nan
        changepoint_data.append([changepoint, care_unit_selected])

    checkpoint_df = pd.DataFrame(changepoint_data, columns={"changepoint", "care_unit"})
    
    # Remove consequtive duplicates, i.e., remove a changepoint if the
    # previous changepoint has the same care unit
    checkpoint_df = checkpoint_df[checkpoint_df.shift(-1)["changepoint"] != checkpoint_df["changepoint"]]
    
    return checkpoint_df


@time_function
def process_care_units(
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
