"""Shared querying functions across use-cases."""

from typing import List, Union

import pandas as pd

import cyclops.query.process as qp
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
)
from cyclops.processors.util import assert_has_columns
from cyclops.query import gemini
from cyclops.query.gemini import get_interface
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from use_cases.gemini.common.constants import ADMIT_VIA_AMBULANCE_MAP, TRIAGE_LEVEL_MAP


def join_queries_flow_fake(
    query_interface_left: Union[QueryInterface, QueryInterfaceProcessed],
    query_interface_right: Union[QueryInterface, QueryInterfaceProcessed],
    **pd_merge_kwargs
) -> pd.DataFrame:
    """Temporary stand-in for the existing join queries workflow to avoid overhead.

    Parameters
    ----------
    query_interface_left: cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        Query acting as the 'left' table in the join.
    query_interface_right: cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        Query acting as the 'right' table in the join.
    **pd_merge_kwargs
        Keyword arguments used in pandas.merge.

    Returns
    -------
    pandas.DataFrame
        The joined data.

    """
    return pd.merge(
        query_interface_left.run(), query_interface_right.run(), **pd_merge_kwargs
    )


@assert_has_columns(ENCOUNTER_ID)
def get_er_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get ER data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    table = gemini.er_admin().query
    table = qp.FilterColumns([ENCOUNTER_ID, "admit_via_ambulance", "triage_level"])(
        table
    )
    table = get_interface(table).run()

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)

    # Map admit_via_ambulance
    # concatenate the combined and air categories together
    for key, value in ADMIT_VIA_AMBULANCE_MAP.items():
        cohort["admit_via_ambulance"] = cohort["admit_via_ambulance"].replace(
            value, key
        )

    for key, value in TRIAGE_LEVEL_MAP.items():
        cohort["triage_level"] = cohort["triage_level"].replace(value, key)

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_derived_variables_for_cohort(
    cohort: pd.DataFrame, derived_variables: List[str]
) -> pd.DataFrame:
    """Get derived variables for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.
    derived_variables: list
        List of derived variables to get.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    table = gemini.derived_variables(variables=derived_variables).run()

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_labs_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get lab data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Lab data for the cohort.

    """
    table = gemini.events(
        "lab", drop_null_event_names=True, drop_null_event_values=True
    ).query
    table = qp.FilterColumns(
        [ENCOUNTER_ID, EVENT_NAME, EVENT_VALUE, EVENT_VALUE_UNIT, EVENT_TIMESTAMP]
    )(table)

    table = get_interface(table).run()
    table[EVENT_CATEGORY] = "labs"

    # Merge with cohort
    return cohort[[ENCOUNTER_ID]].merge(table, on=ENCOUNTER_ID)
