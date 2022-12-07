"""GEMINI mortality decompensation use case querying."""

# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file

import pandas as pd
from drift_detection.gemini.mortality.constants import BEFORE_DATE, OUTCOME_DEATH, SEXES

import cyclops.query.process as qp
from cyclops.process.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DIAGNOSIS_CODE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    HOSPITAL_ID,
    SEX,
    SUBJECT_ID,
)
from cyclops.process.diagnoses import process_diagnoses
from cyclops.process.util import assert_has_columns
from cyclops.query import gemini
from cyclops.query.gemini import get_interface

# from use_cases.gemini.common.constants import READMISSION_MAP
# from use_cases.gemini.common.query import (
#     get_er_for_cohort,
#     get_labs_for_cohort,
#     join_queries_flow_fake,
# )


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


def get_diagnoses() -> pd.DataFrame:
    """Get diagnoses.

    Returns
    -------
    pandas.DataFrame
        The table.

    """
    table = gemini.diagnoses(diagnosis_types="M").query

    # Drop ER diagnoses
    table = qp.ConditionEquals("is_er_diagnosis", False)(table)

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
    diagnoses = get_diagnoses()
    cohort = pd.merge(encounters, diagnoses, on=ENCOUNTER_ID)

    # Include diagnosis code groupings (trajectories)
    trajectory = process_diagnoses(cohort[DIAGNOSIS_CODE])
    cohort[trajectory.name] = trajectory

    return cohort


@assert_has_columns(ENCOUNTER_ID)
def get_imaging_for_cohort(cohort: pd.DataFrame):
    """Get imaging data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Imaging data for cohort.

    """
    imaging = gemini.imaging().run()
    imaging = imaging.rename(
        columns={
            "imaging_test_description": EVENT_NAME,
            "performed_date_time": EVENT_TIMESTAMP,
        }
    )
    imaging[EVENT_VALUE] = 1
    imaging[EVENT_CATEGORY] = "imaging"
    imaging = imaging.loc[imaging[ENCOUNTER_ID].isin(cohort[ENCOUNTER_ID])]
    imaging = imaging[
        [ENCOUNTER_ID, EVENT_NAME, EVENT_CATEGORY, EVENT_VALUE, EVENT_TIMESTAMP]
    ]

    return imaging


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
        Blood transfusions data for cohort.

    """
    transfusions = gemini.blood_transfusions().run()
    transfusions = transfusions.rename(columns={"issue_date_time": EVENT_TIMESTAMP})
    transfusions[EVENT_NAME] = transfusions["rbc_mapped"]
    transfusions[EVENT_NAME] = transfusions[EVENT_NAME].apply(
        lambda x: "rbc" if x else "non-rbc"
    )
    transfusions[EVENT_VALUE] = 1
    transfusions[EVENT_CATEGORY] = "transfusions"
    transfusions = transfusions.loc[
        transfusions[ENCOUNTER_ID].isin(cohort[ENCOUNTER_ID])
    ]
    transfusions = transfusions[
        [ENCOUNTER_ID, EVENT_NAME, EVENT_CATEGORY, EVENT_VALUE, EVENT_TIMESTAMP]
    ]

    return transfusions


@assert_has_columns(ENCOUNTER_ID)
def get_interventions_for_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get interventions data for the cohort.

    Parameters
    ----------
    cohort: pandas.DataFrame
        Cohort data.

    Returns
    -------
    pandas.DataFrame
        Interventions data for cohort.

    """
    interventions = gemini.interventions().run()
    interventions = interventions.loc[
        interventions[ENCOUNTER_ID].isin(cohort[ENCOUNTER_ID])
    ].copy()
    interventions[EVENT_VALUE] = 1
    interventions[EVENT_CATEGORY] = "interventions"
    binary_mapped_cols = [
        "endoscopy_mapped",
        "gi_endoscopy_mapped",
        "bronch_endoscopy_mapped",
        "dialysis_mapped",
        "inv_mech_vent_mapped",
        "surgery_mapped",
    ]
    interventions = interventions[
        ~interventions["intervention_episode_start_date"].isna()
    ]
    interventions["intervention_episode_start_time"].loc[
        interventions["intervention_episode_start_time"].isna()
    ] = "12:00:00"
    interventions[EVENT_TIMESTAMP] = pd.to_datetime(
        interventions["intervention_episode_start_date"].astype(str)
        + " "
        + interventions["intervention_episode_start_time"].astype(str)
    )
    interventions[EVENT_TIMESTAMP] = interventions[EVENT_TIMESTAMP].astype(
        "datetime64[ns]"
    )
    interventions["unmapped_intervention"] = ~(
        interventions["endoscopy_mapped"]
        | interventions["gi_endoscopy_mapped"]
        | interventions["bronch_endoscopy_mapped"]
        | interventions["dialysis_mapped"]
        | interventions["inv_mech_vent_mapped"]
        | interventions["surgery_mapped"]
    )
    interventions[EVENT_NAME] = interventions[
        binary_mapped_cols + ["unmapped_intervention"]
    ].idxmax(axis=1)
    interventions = interventions[
        [ENCOUNTER_ID, EVENT_NAME, EVENT_CATEGORY, EVENT_VALUE, EVENT_TIMESTAMP]
    ]

    return interventions


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

    if drop_admin_cols:
        cohort = cohort.drop(
            ["subject_id", "ccsr_default", "ccsr_1", "ccsr_2"],
            axis=1,
        )

    # Get blood transfusion data for the cohort
    transfusions = get_bt_for_cohort(cohort)

    # Get imaging data for the cohort
    imaging = get_imaging_for_cohort(cohort)

    # Get interventions data for the cohort
    interventions = get_interventions_for_cohort(cohort)

    # Get lab data for the cohort
    labs = get_labs_for_cohort(cohort)

    # Combine events
    events = combine_events([labs, transfusions, imaging, interventions])
    events["event_value"] = events["event_value"].astype(str)

    return cohort, events


if __name__ == "__main__":
    main()
