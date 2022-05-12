"""Process functions applied to queried data."""

from cyclops.processors.column_names import CARE_UNIT
from cyclops.query.mimic import CARE_UNIT_MAP


def process_mimic_care_unit(transfers, specific=False):
    """Process care unit data.

    Processes the MIMIC Transfers table into a cleaned and simplified care
    unit DataFrame.

    Parameters
    ----------
    transfers : pandas.DataFrame
        MIMIC transfers table as a DataFrame.
    specific : bool
        Whether care_unit_name column has specific or non-specific care units.

    Returns
    -------
    pandas.DataFrame
        Processed care unit DataFrame.

    """
    transfers.rename(
        columns={
            "intime": "admit",
            "outtime": "discharge",
            "careunit": CARE_UNIT,
        },
        inplace=True,
    )

    # Drop rows with eventtype discharge
    # Its admit timestamp is the discharge timestamp of eventtype admit
    transfers = transfers[transfers["eventtype"] != "discharge"]

    transfers.drop("eventtype", axis=1, inplace=True)
    transfers = transfers[transfers[CARE_UNIT] != "Unknown"]

    # Create replacement dictionary for care unit categories depending on specificity
    replace_dict = {}
    for unit, unit_dict in CARE_UNIT_MAP.items():
        for specific_unit, unit_list in unit_dict.items():
            value = specific_unit if specific else unit
            replace_dict.update({elem: value for elem in unit_list})

    transfers[CARE_UNIT].replace(replace_dict, inplace=True)

    transfers.dropna(inplace=True)
    return transfers
