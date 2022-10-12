"""Shared querying functions across use-cases."""

from typing import Optional, Union

import numpy as np
import pandas as pd

import cyclops.query.process as qp
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
)
from cyclops.processors.util import assert_has_columns
from cyclops.query import gemini
from cyclops.query.gemini import get_interface
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from use_cases.gemini.common.constants import (
    ADMIT_VIA_AMBULANCE_MAP,
    BT_SUBSTRINGS,
    DERIVED_VARIABLES,
    EDEMA_IMAGING_SUBSTRINGS,
    EDEMA_PHARMA_SUBSTRINGS,
    IMAGING_DESCRIPTIONS,
    IMAGING_KEYWORDS,
    OUTCOME_EDEMA,
    PRESCRIPTION_AFTER_IMAGING_DAYS,
    TRIAGE_LEVEL_MAP,
)


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
def get_bt_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get blood transfusion data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    bt_names = ["bt_" + sub for sub in BT_SUBSTRINGS]

    # DON'T FORGET TO ADD "before_date" FOR COVID!
    table = gemini.blood_transfusions(
        blood_product_raw_substring=BT_SUBSTRINGS,
    ).query

    table = qp.Rename({"rbc_mapped": "bt_rbc"})(table)

    for i, sub in enumerate(BT_SUBSTRINGS):
        table = qp.ConditionSubstring(
            "blood_product_raw", sub, binarize_col=bt_names[i]
        )(table)

    bt_names = bt_names + ["bt_rbc"]

    # Cast boolean columns to 0-1 integer columns
    table = qp.Cast(bt_names, "int")(table)

    # Sum number of each transfusion for a given encounter
    aggfuncs = {}
    for name in bt_names:
        aggfuncs[name] = "sum"

    table = qp.GroupByAggregate(ENCOUNTER_ID, aggfuncs)(table)

    table = get_interface(table).run()

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)

    for name in bt_names:
        cohort[name] = cohort[name].fillna(0).astype(int)

    return cohort


def _imaging_postprocess(
    table: pd.DataFrame, num_tests_thresh: Optional[int] = None
) -> pd.DataFrame:
    """Postprocess the imaging table.

    Parameters
    ----------
    table: pandas.DataFrame
        The imaging table.
    num_tests_thresh: int, optional
        Threshold for the number of tests that must be ordered for a column to be kept.

    Returns
    -------
    pandas.DataFrame
        Processed data.

    """
    # Classify each body part and imaging description into their own categories.
    resulting_cols = []
    for body_part in list(IMAGING_KEYWORDS.keys()):
        for description in IMAGING_DESCRIPTIONS:
            col = description + "_" + body_part
            table[col] = np.where(
                (table[body_part] == 1)
                & (table["imaging_test_description"] == description),
                1,
                0,
            )
            resulting_cols.append(col)

    aggfuncs = {col: "sum" for col in resulting_cols}
    table = table.groupby(ENCOUNTER_ID).agg(aggfuncs).reset_index()

    # Remove any columns with few images across all encounters, e.g.,
    # a whole-body ultrasound has 0 occurences
    if num_tests_thresh is not None:
        for col in set(table.columns) - set([ENCOUNTER_ID]):
            if table[col].sum() < num_tests_thresh:
                table = table.drop(col, axis=1)

    return table


@assert_has_columns(ENCOUNTER_ID)
def get_imaging_for_cohort(
    cohort: pd.DataFrame, num_tests_thresh: Optional[int] = None
):
    """Get imaging data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.
    num_tests_thresh: int, optional
        Threshold for the number of tests that must be ordered for a column to be kept.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    table = gemini.imaging(test_descriptions=IMAGING_DESCRIPTIONS).query

    for key, keywords in IMAGING_KEYWORDS.items():
        table = qp.ConditionSubstring(
            "imaging_test_name_raw", keywords, binarize_col=key
        )(table)
        table = qp.Cast(key, "int")(table)

    table = gemini.get_interface(
        table,
        process_fn=lambda x: _imaging_postprocess(x, num_tests_thresh=num_tests_thresh),
    ).run()

    image_count_cols = set(table.columns) - set([ENCOUNTER_ID])

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)

    for col in image_count_cols:
        cohort[col] = cohort[col].fillna(0).astype(int)

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_derived_variables_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get derived variables for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    table = gemini.derived_variables(variables=DERIVED_VARIABLES).run()

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)

    return cohort


# @assert_has_columns(ENCOUNTER_ID)
def get_pulmonary_edema_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get pulmonary edema indicator for the cohort using imaging and pharmacy data.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    imaging = gemini.imaging().query

    # Note: Imaging results must include all of the substrings.
    imaging = qp.FilterColumns([ENCOUNTER_ID, "test_result", "performed_date_time"])(
        imaging
    )
    imaging = qp.ConditionSubstring(
        "test_result",
        EDEMA_IMAGING_SUBSTRINGS,
        any_=False,
    )(imaging)
    imaging = get_interface(imaging).run()

    # Pharmacy data
    pharma = gemini.pharmacy().query

    # Note: Medication name must include any of the substrings.
    pharma = qp.FilterColumns(
        [ENCOUNTER_ID, "med_id_generic_name_raw", "med_order_start_date_time"]
    )(pharma)
    pharma = qp.ConditionSubstring(
        "med_id_generic_name_raw",
        EDEMA_PHARMA_SUBSTRINGS,
        # binarize_col="edema_pharma",
    )(pharma)
    pharma = get_interface(pharma).run()

    # Create the pulmonary edema indicator when an imaging test is performed
    # with the proper substrings, and then a certain medication is prescribed
    # within a certain period of time
    imaging_pharma = imaging.merge(pharma, on=ENCOUNTER_ID)

    imaging_pharma["prescribed_after"] = pd.to_datetime(
        imaging_pharma["med_order_start_date_time"], errors="coerce"
    ) - pd.to_datetime(imaging_pharma["performed_date_time"], errors="coerce")
    imaging_pharma[OUTCOME_EDEMA] = (
        imaging_pharma["prescribed_after"].dt.days >= 0
    ) & (imaging_pharma["prescribed_after"].dt.days <= PRESCRIPTION_AFTER_IMAGING_DAYS)
    imaging_pharma = imaging_pharma[imaging_pharma[OUTCOME_EDEMA]]

    imaging_pharma = imaging_pharma[[ENCOUNTER_ID, OUTCOME_EDEMA]].drop_duplicates()
    cohort = cohort.merge(imaging_pharma, how="left", on=ENCOUNTER_ID)
    cohort[OUTCOME_EDEMA] = cohort[OUTCOME_EDEMA].fillna(False)
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

    # Merge with cohort
    return cohort[[ENCOUNTER_ID]].merge(table, on=ENCOUNTER_ID)