"""WangLab cardiac use case querying."""

import pandas as pd

import cyclops.query.process as qp
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DIAGNOSIS_CODE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    HOSPITAL_ID,
    SEX,
    SUBJECT_ID,
)
from cyclops.processors.diagnoses import process_diagnoses
from cyclops.query import gemini
from cyclops.query.gemini import get_interface
from use_cases.gemini.common.constants import OUTCOME_DEATH, READMISSION_MAP
from use_cases.gemini.common.query import (
    get_bt_for_cohort,
    get_derived_variables_for_cohort,
    get_er_for_cohort,
    get_imaging_for_cohort,
    get_labs_for_cohort,
    get_pulmonary_edema_for_cohort,
    join_queries_flow_fake,
)
from use_cases.gemini.wanglab.constants import BEFORE_DATE, SEXES


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
        died_binarize_col=OUTCOME_DEATH,
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
        OUTCOME_DEATH,
        "readmission",
        "from_nursing_home_mapped",
        "from_acute_care_institution_mapped",
        "los_derived",
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
    cohort = pd.merge(encounters, diagnoses, on=ENCOUNTER_ID)

    # Include diagnosis code groupings (trajectories)
    trajectory = process_diagnoses(cohort[DIAGNOSIS_CODE])
    cohort[trajectory.name] = trajectory

    return cohort


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
    cohort = get_derived_variables_for_cohort(cohort)

    # Get pulmonary edema indicator for the cohort
    cohort = get_pulmonary_edema_for_cohort(cohort)

    if drop_admin_cols:
        cohort = cohort.drop(
            ["subject_id", "ccsr_default", "ccsr_1", "ccsr_2"],
            axis=1,
        )

    # Get lab data
    labs = get_labs_for_cohort(cohort)

    return cohort, labs


if __name__ == "__main__":
    main()
