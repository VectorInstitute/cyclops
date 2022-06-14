"""WangLab cardiac use case querying."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from wanglab_constants import (
    ADMIT_VIA_AMBULANCE_MAP,
    BEFORE_DATE,
    BT_SUBSTRINGS,
    DERIVED_VARIABLES,
    EDEMA_IMAGING_SUBSTRINGS,
    EDEMA_PHARMA_SUBSTRINGS,
    IMAGING_DESCRIPTIONS,
    IMAGING_KEYWORDS,
    READMISSION_MAP,
    SEXES,
    TRIAGE_LEVEL_MAP,
)

import cyclops.query.process as qp
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DIAGNOSIS_CODE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    HOSPITAL_ID,
    SEX,
    SUBJECT_ID,
)
from cyclops.processors.util import assert_has_columns
from cyclops.query import gemini
from cyclops.query.gemini import get_interface
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed


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


def get_most_recent_encounters() -> pd.DataFrame:
    """Filter for cohort and get the most recent encounter for each patient.

    Returns
    -------
    pandas.DataFrame
        The recent patient encounters.

    """
    table = gemini.patient_encounters(
        sex=SEXES,
        before_date=BEFORE_DATE,
        died=True,
        died_binarize_col="outcome_death",
    ).query

    # Do not do any further filtering before this point since
    # we count previous encounters below.

    # Get most recent admission for each patient
    recent_admits = qp.GroupByAggregate(
        SUBJECT_ID,
        {ADMIT_TIMESTAMP: "max", SUBJECT_ID: ("count", "prev_encounter_count")},
    )(table)

    # Subtract one from encounter count to get the count of count previous encounters
    recent_admits = qp.AddNumeric("prev_encounter_count", -1)(recent_admits)

    # Only keep encounters where most responsible physician is GIM
    table = qp.ConditionEquals("mrp_gim", "y")(table)

    # Filter columns
    keep = [
        ENCOUNTER_ID,
        SUBJECT_ID,
        ADMIT_TIMESTAMP,
        DISCHARGE_TIMESTAMP,
        AGE,
        SEX,
        HOSPITAL_ID,
        "outcome_death",
        "readmission",
        "from_nursing_home_mapped",
        "from_acute_care_institution_mapped",
        "los_derived",
        # "admit_category",
        # "institution_from_type",
    ]
    table = qp.FilterColumns(keep)(table)

    table = qp.ReorderAfter(ADMIT_TIMESTAMP, SUBJECT_ID)(table)

    # Keep only most recent encounter
    cohort = join_queries_flow_fake(
        get_interface(table),
        get_interface(recent_admits),
        on=[SUBJECT_ID, ADMIT_TIMESTAMP],
    )

    for key, value in READMISSION_MAP.items():
        cohort["readmission"] = cohort["readmission"].replace(value, key)

    return cohort


def get_non_cardiac_diagnoses() -> pd.DataFrame:
    """Get non-cardiac diagnoses.

    Returns
    -------
    pandas.DataFrame
        The table.

    """
    table = gemini.diagnoses(diagnosis_types="M").query

    # Drop ER diagnoses
    table = qp.ConditionEquals("is_er_diagnosis", False)(table)

    # Keep only the encounters with a non-cardiac main diagnosis
    table = qp.ConditionStartsWith("ccsr_1", "CIR", not_=True)(table)

    # Filter columns
    keep = [ENCOUNTER_ID, DIAGNOSIS_CODE, "ccsr_default", "ccsr_1", "ccsr_2"]
    table = qp.FilterColumns(keep)(table)

    return get_interface(table).run()


def get_cohort() -> pd.DataFrame:
    """Get cohort.

    Get cohort of pre-Covid, GIM patients admitted for non-cardiac main diagnoses.

    Returns
    -------
    pandas.DataFrame
        The table.

    """
    encounters = get_most_recent_encounters()
    diagnoses = get_non_cardiac_diagnoses()
    return pd.merge(encounters, diagnoses, on=ENCOUNTER_ID)


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


def imaging_postprocess(
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
        process_fn=lambda x: imaging_postprocess(x, num_tests_thresh=num_tests_thresh),
    ).run()

    image_count_cols = set(table.columns) - set([ENCOUNTER_ID])

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)

    for col in image_count_cols:
        cohort[col] = cohort[col].fillna(0).astype(int)

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_derived_variables(cohort: pd.DataFrame) -> pd.DataFrame:
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


@assert_has_columns(ENCOUNTER_ID)
def pulmonary_edema_imaging(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get pulmonary edema imaging data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    table = gemini.imaging().query

    # Imaging results must include all of the substrings.
    table = qp.ConditionSubstring(
        "test_result",
        EDEMA_IMAGING_SUBSTRINGS,
        any_=False,
        binarize_col="edema_imaging",
    )(table)
    table = qp.Cast("edema_imaging", "int")(table)

    table = qp.GroupByAggregate(ENCOUNTER_ID, {"edema_imaging": "sum"})(table)

    table = get_interface(table).run()

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)
    cohort["edema_imaging"] = cohort["edema_imaging"].fillna(0).astype(int)

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def pulmonary_edema_pharmacy(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get pulmonary edema pharmacy data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    table = gemini.pharmacy().query

    # Medication name must include one of the substrings.
    table = qp.ConditionSubstring(
        "med_id_generic_name_raw", EDEMA_PHARMA_SUBSTRINGS, binarize_col="edema_pharma"
    )(table)
    table = qp.Cast("edema_pharma", "int")(table)

    table = qp.GroupByAggregate(ENCOUNTER_ID, {"edema_pharma": "sum"})(table)

    table = get_interface(table).run()

    # Merge with cohort
    cohort = cohort.merge(table, how="left", on=ENCOUNTER_ID)
    cohort["edema_pharma"] = cohort["edema_pharma"].fillna(0).astype(int)

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_pulmonary_edema_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get pulmonary edema data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Cohort with additional data.

    """
    cohort = pulmonary_edema_imaging(cohort)
    cohort = pulmonary_edema_pharmacy(cohort)
    cohort["outcome_edema"] = (cohort["edema_imaging"] > 0) & (
        cohort["edema_pharma"] > 0
    )
    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_labs(cohort: pd.DataFrame) -> pd.DataFrame:
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


def main(drop_admin_cols=True):
    """Get and process the cohort.

    Parameters
    ----------
    drop_admin_cols: bool, default = True
        Whether to drop the cohort adminstrative columns.

    Returns
    -------
    pandas.DataFrame
        Processed cohort.

    """
    # Get cohort
    cohort = get_cohort()

    # Get ER data for the cohort
    cohort = get_er_for_cohort(cohort)

    # Get blood transfusion data for the cohort
    cohort = get_bt_for_cohort(cohort)

    # Get imaging data for the cohort
    cohort = get_imaging_for_cohort(cohort, num_tests_thresh=5)

    # Get derived variables for the cohort
    cohort = get_derived_variables(cohort)

    # Get pulmonary edema indicator for the cohort
    cohort = get_pulmonary_edema_for_cohort(cohort)

    # Get labs data
    labs = get_labs(cohort)

    if drop_admin_cols:
        cohort = cohort.drop(
            [
                SUBJECT_ID,
                ADMIT_TIMESTAMP,
                DISCHARGE_TIMESTAMP,
                HOSPITAL_ID,
            ],
            axis=1,
        )

    return cohort, labs


if __name__ == "__main__":
    main()
