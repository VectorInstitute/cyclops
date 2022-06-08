"""WangLab cardiac use case querying."""

from typing import Optional

from sqlalchemy.sql.selectable import Subquery

import cyclops.query.process as qp
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    ENCOUNTER_ID,
    HOSPITAL_ID,
    SEX,
    SUBJECT_ID,
)
from cyclops.query import gemini
from cyclops.query.interface import QueryInterface
from cyclops.query.util import (
    TableTypes,
    assert_table_has_columns,
    table_params_to_type,
)


@table_params_to_type(Subquery)
@assert_table_has_columns([ENCOUNTER_ID, SUBJECT_ID, ADMIT_TIMESTAMP])
def get_most_recent_encounter(table: TableTypes) -> QueryInterface:
    """Get the most recent encounter for each patient.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        Table for which to get the most recent encounters.

    Returns
    -------
        The table.

    """
    # First most recent admission
    recent_admits = qp.GroupByAggregate(
        SUBJECT_ID,
        {ADMIT_TIMESTAMP: "max", SUBJECT_ID: ("count", "prev_encounter_count")},
    )(table)

    # Keep only most recent admission, i.e., encounter
    table = qp.Join(
        recent_admits,
        on=[SUBJECT_ID, ADMIT_TIMESTAMP],
        join_table_cols=["prev_encounter_count"],
    )(table)

    # Subtract one from encounter count to get previous encounters count
    table = qp.AddNumeric("prev_encounter_count", -1)(table)

    return gemini.get_interface(table)


def get_encounters(limit: Optional[int] = None) -> QueryInterface:
    """Filter and get the most recent encounter for each patient.

    Parameters
    ----------
    limit: int, optional
        Optionally limit the query for the purposes of debugging.

    Returns
    -------
        The table.

    """
    table = gemini.patient_encounters(
        sex=["M", "F"],
        before_date="2020-01-23",
        died=True,
        died_binarize_col="outcome_death",
    ).query

    # DEBUGGING
    if limit is not None:
        table = qp.Limit(limit)(table)

    # Do not do any further filtering before this point since
    # we count previous encounters in the below function.
    # table = most_recent_encounter(table).query

    return gemini.get_interface(table)


def get_non_cardiac_diagnoses(limit: Optional[int] = None) -> QueryInterface:
    """Get non-cardiac diagnoses.

    Parameters
    ----------
    limit: int, optional
        Optionally limit the query for the purposes of debugging.

    Returns
    -------
        The table.

    """
    table = gemini.diagnoses(diagnosis_types="M").query

    # DEBUGGING
    if limit is not None:
        table = qp.Limit(limit)(table)

    table = qp.Drop(["ccsr_3", "ccsr_4", "ccsr_5"])(table)

    # Drop ER diagnoses
    table = qp.ConditionEquals("is_er_diagnosis", False)(table)

    # Keep only the encounters with a non-cardiac main diagnosis
    table = qp.ConditionStartsWith("ccsr_1", "CIR", not_=True)(table)

    return gemini.get_interface(table)


def get_cohort(limit: Optional[int] = None) -> QueryInterface:
    """Get cohort.

    Get cohort of pre-Covid, GIM patients admitted for
    non-cardiac main diagnoses.

    Parameters
    ----------
    limit: int, optional
        Optionally limit the query for the purposes of debugging.

    Returns
    -------
        The table.

    """
    table = get_encounters(limit=limit).query

    # Only keep encounters where most responsible physician is GIM
    table = qp.ConditionEquals("mrp_gim", "y")(table)

    # Filter columns
    keep = [
        ENCOUNTER_ID,
        SUBJECT_ID,
        ADMIT_TIMESTAMP,
        AGE,
        SEX,
        HOSPITAL_ID,
        "outcome_death",
        "admit_category",
        "readmission",
        "institution_from_type",
        "from_nursing_home_mapped",
        "from_acute_care_institution_mapped",
    ]
    table = qp.FilterColumns(keep)(table)

    # Remove null SUBJECT_ID
    table = qp.DropNulls(SUBJECT_ID)(table)

    table = qp.ReorderAfter(ADMIT_TIMESTAMP, SUBJECT_ID)(table)

    diagnoses = get_non_cardiac_diagnoses(limit=limit).query

    table = qp.Join(diagnoses, on=ENCOUNTER_ID)(table)

    return gemini.get_interface(table)
