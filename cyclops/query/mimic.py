"""MIMIC-IV query API."""

# pylint: disable=duplicate-code

import logging
from typing import List, Optional

from sqlalchemy import Integer, func, select
from sqlalchemy.sql.selectable import Subquery

from codebase_ops import get_log_file_path
from cyclops import config
from cyclops.constants import MIMIC
from cyclops.orm import Database
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DATE_OF_DEATH,
    DIAGNOSIS_CODE,
    DIAGNOSIS_TITLE,
    DIAGNOSIS_VERSION,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    SEX,
    SUBJECT_ID,
)
from cyclops.query import process as qp
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.postprocess.mimic import process_mimic_care_units
from cyclops.query.util import (
    QueryTypes,
    _to_subquery,
    assert_query_has_columns,
    get_attribute,
    query_params_to_type,
)
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


PATIENTS = "patients"
ADMISSIONS = "admissions"
DIAGNOSES = "diagnoses"
PATIENT_DIAGNOSES = "patient_diagnoses"
EVENT_LABELS = "event_labels"
EVENTS = "events"
TRANSFERS = "transfers"
ED_STAYS = "ed_stays"

_db = Database(config.read_config(MIMIC))
TABLE_MAP = {
    PATIENTS: lambda db: db.mimic_core.patients,
    ADMISSIONS: lambda db: db.mimic_core.admissions,
    DIAGNOSES: lambda db: db.mimic_hosp.d_icd_diagnoses,
    PATIENT_DIAGNOSES: lambda db: db.mimic_hosp.diagnoses_icd,
    EVENT_LABELS: lambda db: db.mimic_icu.d_items,
    EVENTS: lambda db: db.mimic_icu.chartevents,
    TRANSFERS: lambda db: db.mimic_core.transfers,
    ED_STAYS: lambda db: db.mimic_ed.edstays,
}
MIMIC_COLUMN_MAP = {
    "hadm_id": ENCOUNTER_ID,
    "subject_id": SUBJECT_ID,
    "admittime": ADMIT_TIMESTAMP,
    "dischtime": DISCHARGE_TIMESTAMP,
    "gender": SEX,
    "anchor_age": AGE,
    "valueuom": EVENT_VALUE_UNIT,
    "label": EVENT_NAME,
    "valuenum": EVENT_VALUE,
    "charttime": EVENT_TIMESTAMP,
    "icd_code": DIAGNOSIS_CODE,
    "icd_version": DIAGNOSIS_VERSION,
    "dod": DATE_OF_DEATH,
}


def get_table(table_name: str, rename: bool = True) -> Subquery:
    """Get a table and possibly map columns to have standard names.

    Standardizing column names allows for for columns to be
    recognized in downstream processing.

    Parameters
    ----------
    table_name: str
        Name of MIMIC table.
    rename: bool, optional
        Whether to map the column names

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Query with mapped columns.

    """
    if table_name not in TABLE_MAP:
        raise ValueError(f"{table_name} not a recognised table.")

    table = TABLE_MAP[table_name](_db).data

    if rename:
        table = qp.Rename(MIMIC_COLUMN_MAP, check_exists=False)(table)

    return _to_subquery(table)


def patients(**process_kwargs) -> QueryInterface:
    """Query MIMIC patient data.

    Other Parameters
    ----------------
    sex: str or list of string, optional
        Specify patient sex (one or multiple).
    died: bool, optional
        Specify True to get patients who have died, and False for those who haven't.
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(PATIENTS)

    # Process and include patient's anchor year.
    table = select(
        table,
        (
            func.substr(get_attribute(table, "anchor_year_group"), 1, 4).cast(
                Integer
            )
        ).label("anchor_year_group_start"),
        (
            func.substr(get_attribute(table, "anchor_year_group"), 8, 12).cast(
                Integer
            )
        ).label("anchor_year_group_end"),
    ).subquery()

    # Select the middle of the anchor year group as the anchor year
    table = select(
        table,
        (
            get_attribute(table, "anchor_year_group_start")
            + (
                get_attribute(table, "anchor_year_group_end")
                - get_attribute(table, "anchor_year_group_start")
            )
            / 2
        ).label("anchor_year_group_middle"),
    ).subquery()

    table = select(
        table,
        (
            get_attribute(table, "anchor_year_group_middle")
            - get_attribute(table, "anchor_year")
        ).label("anchor_year_difference"),
    ).subquery()

    # Shift relevant columns by anchor year difference
    table = qp.AddColumn("anchor_year", "anchor_year_difference")(
        table
    )
    table = qp.AddDeltaColumns(DATE_OF_DEATH, years="anchor_year_difference")(
        table
    )

    # Calculate approximate year of birth
    table = qp.AddColumn(
        "anchor_year", "age", negative=True, new_col_labels="birth_year"
    )(table)

    table = qp.Drop(
        [
            "age",
            "anchor_year",
            "anchor_year_group",
            "anchor_year_group_start",
            "anchor_year_group_end",
            "anchor_year_group_middle",
        ]
    )(table)

    # Reorder nicely.
    table = qp.Reorder(
        [SUBJECT_ID, SEX, "birth_year", DATE_OF_DEATH, "anchor_year_difference"]
    )(table)

    # Process optional operations
    operations.extend([
        # Must convert to string since CHAR(1) type doesn't recognize equality
        (qp.ConditionIn, [SEX, qp.QAP("sex")], {"to_str": True}),
        (qp.ConditionEquals, [DATE_OF_DEATH, None], {"not_": qp.QAP(process_kwargs["died"], not_=True)})
        (qp.Limit, [qp.QAP("limit")], {}),
    ])

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


def diagnoses(**process_kwargs) -> QueryInterface:
    """Query MIMIC diagnoses.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    diagnosis_versions: int or list of int, optional
        Get codes having certain ICD versions.
    diagnosis_substring : str, optional
        Substring to match in the ICD code.
    diagnosis_codes : str or list of str, optional
        Get only the specified ICD codes.
    limit: int, optional
        Limit the number of rows returned.
    """
    table = get_table(DIAGNOSES)

    # Rename long_title
    table = qp.Rename({"long_title": DIAGNOSIS_TITLE})(table)

    # Trim whitespace from ICD codes.
    table = qp.Trim(DIAGNOSIS_CODE)(table)

    # Process optional operations
    operations = [
        (
            qp.ConditionIn,
            [DIAGNOSIS_VERSION, qp.QAP("diagnosis_versions")],
            {"to_int": True},
        ),
        (qp.ConditionSubstring, [DIAGNOSIS_TITLE, qp.QAP("diagnosis_substring")], {}),
        (qp.ConditionIn, [DIAGNOSIS_CODE, qp.QAP("diagnosis_codes")], {"to_str": True}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@query_params_to_type(Subquery)
@assert_query_has_columns(patients_table=SUBJECT_ID)
def patient_diagnoses(
    patients_table: Optional[QueryTypes] = None, **process_kwargs
) -> QueryInterface:
    """Query MIMIC patient diagnoses.

    Parameters
    ----------
    patients: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable, optional
        Patient encounters query used to join.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    diagnosis_versions: int or list of int, optional
        Get codes having certain ICD versions.
    diagnosis_substring: str, optional
        Substring to match in the ICD code.
    diagnosis_codes: str or list of str, optional
        Get only the specified ICD codes.

    """
    # Get patient diagnoses.
    table = get_table(PATIENT_DIAGNOSES)

    # Trim whitespace from ICD codes.
    table = qp.Trim(DIAGNOSIS_CODE)(table)

    # If provided, join with a patients table
    if patients_table is not None:
        table = qp.Join(patients_table, on=SUBJECT_ID)(table)

    # Get diagnosis codes.
    diagnoses_table = diagnoses(
        diagnosis_versions=qp.ckwarg(process_kwargs, "diagnosis_versions"),
        diagnosis_substring=qp.ckwarg(process_kwargs, "diagnosis_substring"),
        diagnosis_codes=qp.ckwarg(process_kwargs, "diagnosis_codes"),
    ).query
    process_kwargs = qp.remove_kwargs(
        process_kwargs, ["diagnosis_versions", "diagnosis_substring", "diagnosis_codes"]
    )

    # Include DIAGNOSIS_TITLE in patient diagnoses.
    table = qp.Join(
        diagnoses_table,
        on=[DIAGNOSIS_CODE, DIAGNOSIS_VERSION],
        join_table_cols=DIAGNOSIS_TITLE,
    )(table)

    return QueryInterface(_db, table)


@query_params_to_type(Subquery)
@assert_query_has_columns(patients_table=SUBJECT_ID)
def transfers(
    patients_table: Optional[QueryTypes] = None, **process_kwargs
) -> QueryInterfaceProcessed:
    """Get care unit table within a given set of encounters.

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select
    or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table
    or cyclops.query.utils.DBTable, optional
        Patient encounters used to join.

    Returns
    -------
    cyclops.query.interface.QueryInterfaceProcessed
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    encounters : list, optional
        The encounter IDs on which to filter. If None, consider all encounters.
    limit: int, optional
        Limit the number of rows returned.
    """
    table = get_table(TRANSFERS)

    if patients_table is not None:
        table = qp.Join(patients_table, on=SUBJECT_ID)(table)

        table = qp.AddDeltaColumns(
            ["intime", "outtime"], years="anchor_year_difference"
        )(table)

    # Process optional operations
    operations = [
        (qp.ConditionIn, [ENCOUNTER_ID, qp.QAP("encounters")], {"to_int": True}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@query_params_to_type(Subquery)
@assert_query_has_columns(patients_table=SUBJECT_ID)
def care_units(
    patients_table: Optional[QueryTypes] = None, **process_kwargs
) -> QueryInterfaceProcessed:
    """Get care unit table within a given set of encounters.

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select
    or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table
    or cyclops.query.utils.DBTable, optional
        Patient encounters to join with transfers.

    Returns
    -------
    cyclops.query.interface.QueryInterfaceProcessed
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    encounters : int or list of int, optional
        Get the specific encounter IDs.

    """
    table = transfers(
        patients_table=patients_table,
        encounters=qp.ckwarg(process_kwargs, "encounters"),
    ).query
    process_kwargs = qp.remove_kwargs(process_kwargs, "encounters")

    return QueryInterfaceProcessed(
        _db,
        table,
        process_fn=process_mimic_care_units,
        process_fn_kwargs={"specific": False},
    )


@query_params_to_type(Subquery)
@assert_query_has_columns(patients_table=SUBJECT_ID)
def patient_encounters(
    patients_table: Optional[QueryTypes] = None, **process_kwargs
) -> QueryInterface:
    """Query MIMIC patient encounters.

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table
    or cyclops.query.utils.DBTable, optional
        Optionally provide a patient table when getting patient encounters.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    before_date: datetime.datetime or str
        Get patients encounters before some date. If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patients encounters after some date. If a string, provide in YYYY-MM-DD format.
    years: int or list of int, optional
        Get patient encounters by year.
    months: int or list of int, optional
        Get patient encounters by month.

    """
    table = get_table(ADMISSIONS)

    if patients_table is None:
        patients_table = patients().query

    # Join admissions and patient table
    table = qp.Join(patients_table, on="subject_id")(table)

    # Update timestamps with anchor year difference
    table = qp.AddDeltaColumns(
        [ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP], years="anchor_year_difference"
    )(table)

    # Process optional operations
    operations: List[tuple] = [
        (qp.ConditionBeforeDate, ["admit_timestamp", qp.QAP("before_date")], {}),
        (qp.ConditionAfterDate, ["admit_timestamp", qp.QAP("after_date")], {}),
        (qp.ConditionInYears, ["admit_timestamp", qp.QAP("years")], {}),
        (qp.ConditionInMonths, ["admit_timestamp", qp.QAP("months")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@query_params_to_type(Subquery)
@assert_query_has_columns(patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID])
def events(
    patient_encounters_table: Optional[QueryTypes] = None, **process_kwargs
) -> QueryInterface:
    """Query MIMIC events.

    Parameters
    ----------
    patient_encounters_table: sqlalchemy.sql.selectable.Select or
    or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table
    or cyclops.query.utils.DBTable, optional
        Optionally provide a patient encounter table to join with events.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    categories: str or list of str, optional
        Restrict to certain categories.
    event_names: str or list of str, optional
        Restrict to certain event names.
    event_name_substring: str, optional
        Substring to search event names to filter.
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(EVENTS)
    event_labels = get_table(EVENT_LABELS)

    table = qp.Join(
        event_labels, on="itemid", join_table_cols=["category", "event_name"]
    )(table)

    # Process optional operations
    operations: List[tuple] = [
        (qp.ConditionIn, ["category", qp.QAP("categories")], {}),
        (qp.ConditionIn, ["event_name", qp.QAP("event_names")], {}),
        (qp.ConditionSubstring, ["event_name", qp.QAP("event_name_substring")], {}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    # Join on patient encounters
    if patient_encounters_table is not None:
        table = qp.Join(
            patient_encounters_table,
            on=ENCOUNTER_ID,
        )(table)

        # Add MIMIC patient-specific time difference to event/store timestamps
        table = qp.AddDeltaColumns(
            [EVENT_TIMESTAMP, "storetime"], years="anchor_year_difference"
        )(table)

    return QueryInterface(_db, table)
