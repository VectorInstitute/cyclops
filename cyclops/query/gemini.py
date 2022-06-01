"""GEMINI query API."""

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.sql.expression import union_all
from sqlalchemy.sql.selectable import Subquery

from codebase_ops import get_log_file_path
from cyclops import config
from cyclops.constants import GEMINI
from cyclops.orm import Database
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    CARE_UNIT,
    DIAGNOSIS_CODE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    ER_ADMIT_TIMESTAMP,
    ER_DISCHARGE_TIMESTAMP,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    HOSPITAL_ID,
    LENGTH_OF_STAY_IN_ER,
    SCU_ADMIT_TIMESTAMP,
    SCU_DISCHARGE_TIMESTAMP,
    SEX,
    SUBJECT_ID,
)
from cyclops.processors.constants import EMPTY_STRING
from cyclops.query import process as qp
from cyclops.query.interface import QueryInterface
from cyclops.query.util import (
    TableTypes,
    _to_subquery,
    assert_table_has_columns,
    table_params_to_type,
)
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


IP_ADMIN = "ip_admin"
ER_ADMIN = "er_admin"
DIAGNOSIS = "diagnosis"
LAB = "lab"
VITALS = "vitals"
PHARMACY = "pharmacy"
INTERVENTION = "intervention"
LOOKUP_IP_ADMIN = "lookup_ip_admin"
LOOKUP_ER_ADMIN = "lookup_er_admin"
LOOKUP_DIAGNOSIS = "lookup_diagnosis"
IP_SCU = "ip_scu"
LOOKUP_ROOM_TRANSFER = "lookup_room_transfer"
ROOM_TRANSFER = "room_transfer"

_db = Database(config.read_config(GEMINI))
TABLE_MAP = {
    IP_ADMIN: lambda db: db.public.ip_administrative,
    ER_ADMIN: lambda db: db.public.er_administrative,
    DIAGNOSIS: lambda db: db.public.diagnosis,
    LAB: lambda db: db.public.lab,
    VITALS: lambda db: db.public.vitals,
    PHARMACY: lambda db: db.public.pharmacy,
    INTERVENTION: lambda db: db.public.intervention,
    LOOKUP_IP_ADMIN: lambda db: db.public.lookup_ip_administrative,
    LOOKUP_ER_ADMIN: lambda db: db.public.lookup_er_administrative,
    LOOKUP_DIAGNOSIS: lambda db: db.public.lookup_diagnosis,
    IP_SCU: lambda db: db.public.ip_scu,
    ROOM_TRANSFER: lambda db: db.public.room_transfer,
    LOOKUP_ROOM_TRANSFER: lambda db: db.public.lookup_room_transfer,
}
GEMINI_COLUMN_MAP = {
    "genc_id": ENCOUNTER_ID,
    "admit_date_time": ADMIT_TIMESTAMP,
    "discharge_date_time": DISCHARGE_TIMESTAMP,
    "gender": SEX,
    "result_value": EVENT_VALUE,
    "result_unit": EVENT_VALUE_UNIT,
    "lab_test_name_mapped": EVENT_NAME,
    "sample_collection_date_time": EVENT_TIMESTAMP,
    "measurement_mapped": EVENT_NAME,
    "measurement_value": EVENT_VALUE,
    "measure_date_time": EVENT_TIMESTAMP,
    "left_er_date_time": ER_DISCHARGE_TIMESTAMP,
    "duration_er_stay_derived": LENGTH_OF_STAY_IN_ER,
    "triage_date_time": ER_ADMIT_TIMESTAMP,
    "er_discharge_timestamp": ER_DISCHARGE_TIMESTAMP,
    "scu_admit_date_time": SCU_ADMIT_TIMESTAMP,
    "scu_discharge_date_time": SCU_DISCHARGE_TIMESTAMP,
    "hospital_id": HOSPITAL_ID,
    "diagnosis_code": DIAGNOSIS_CODE,
    "patient_id_hashed": SUBJECT_ID,
}

EVENT_CATEGORIES = [LAB, VITALS]


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
        Table with mapped columns.

    """
    if table_name not in TABLE_MAP:
        raise ValueError(f"{table_name} not a recognised table.")

    table = TABLE_MAP[table_name](_db).data

    if rename:
        table = qp.Rename(GEMINI_COLUMN_MAP, check_exists=False)(table)

    return _to_subquery(table)


def er_admin(**process_kwargs) -> QueryInterface:
    """Query emergency room administrative data.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed table, wrapped in an interface object.

    Other Parameters
    ----------------
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(ER_ADMIN)

    # Process optional operations
    operations: List[tuple] = [(qp.Limit, [qp.QAP("limit")], {})]
    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
@assert_table_has_columns(er_admin_table=ENCOUNTER_ID)
def patient_encounters(
    er_admin_table: Optional[TableTypes] = None, **process_kwargs
) -> QueryInterface:
    """Query GEMINI patient encounters.

    Parameters
    ----------
    er_admin_table: Subquery, optional
        Gather Emergency Room data recorded for the particular encounter.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    sex: str or list of string, optional
        Specify patient sex (one or multiple).
    died: bool, optional
        Specify True to get patients who have died, and False for those who haven't.
    died_binarize_col: str, optional
        Binarize the died condition and save as a column with label died_binarize_col.
    before_date: datetime.datetime or str
        Get patients encounters before some date.
        If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patients encounters after some date.
        If a string, provide in YYYY-MM-DD format.
    hospitals: str or list of str, optional
        Get patient encounters by hospital sites.
    years: int or list of int, optional
        Get patient encounters by year.
    months: int or list of int, optional
        Get patient encounters by month.
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(IP_ADMIN)

    # Get the discharge disposition code descriptions
    lookup_table = get_table(LOOKUP_IP_ADMIN)
    lookup_table = qp.ConditionEquals("variable", "discharge_disposition")(lookup_table)

    table = qp.Join(
        lookup_table,
        on=("discharge_disposition", "value"),
        on_to_type=int,
        join_table_cols="description",
    )(table)
    table = qp.Rename({"description": "discharge_description"})(table)
    table = qp.Drop("value")(table)

    # Join on ER data only if specified
    if er_admin_table is not None:
        table = qp.Join(er_admin_table, on=ENCOUNTER_ID)(table)

    # Process optional operations
    operations: List[tuple] = [
        (qp.ConditionBeforeDate, ["admit_timestamp", qp.QAP("before_date")], {}),
        (qp.ConditionAfterDate, ["admit_timestamp", qp.QAP("after_date")], {}),
        (qp.ConditionInYears, ["admit_timestamp", qp.QAP("years")], {}),
        (qp.ConditionInMonths, ["admit_timestamp", qp.QAP("months")], {}),
        (qp.ConditionIn, [HOSPITAL_ID, qp.QAP("hospitals")], {"to_str": True}),
        (qp.ConditionIn, [SEX, qp.QAP("sex")], {"to_str": True}),
        (
            qp.ConditionEquals,
            ["discharge_description", "Died"],
            {"not_": qp.QAP("died", not_=True)},
        ),
        (
            qp.ConditionEquals,
            ["discharge_description", "Died"],
            {"binarize_col": qp.QAP("died_binarize_col")},
        ),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


def diagnoses(**process_kwargs) -> QueryInterface:
    """Query diagnosis data.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed table, wrapped in an interface object.

    Other Parameters
    ----------------
    diagnosis_codes: str or list of str, optional
        Get only the specified ICD codes.
    diagnosis_types: list of str, optional
        Include only those diagnoses that are of certain type.
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(DIAGNOSIS)

    # Get diagnosis type description
    lookup_diagnosis = get_table(LOOKUP_DIAGNOSIS)
    table = qp.Join(
        lookup_diagnosis, on=("diagnosis_type", "value"), join_table_cols="description"
    )(table)
    table = qp.Drop("value")(table)
    table = qp.Rename({"description": "diagnosis_type_description"})(table)
    table = qp.ReorderAfter("diagnosis_type_description", "diagnosis_type")(table)

    # Trim whitespace from ICD codes.
    table = qp.Trim(DIAGNOSIS_CODE)(table)

    # Process optional operations
    operations: List[tuple] = [
        (qp.ConditionIn, [DIAGNOSIS_CODE, qp.QAP("diagnosis_codes")], {"to_str": True}),
        (
            qp.ConditionIn,
            ["diagnosis_type", qp.QAP("diagnosis_types")],
            {"to_str": True},
        ),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]
    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
@assert_table_has_columns(
    diagnoses_table=[ENCOUNTER_ID, DIAGNOSIS_CODE],
    patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID],
)
def patient_diagnoses(
    diagnoses_table: Optional[TableTypes] = None,
    patient_encounters_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query diagnosis data.

    Parameters
    ----------
    diagnoses_table: cyclops.query.util.TableTypes, optional
        Diagnoses table used to join.
    patient_encounters_table: cyclops.query.util.TableTypes, optional
        Patient encounters table used to join.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed table, wrapped in an interface object.

    Other Parameters
    ----------------
    diagnosis_codes: str or list of str, optional
        Get only the specified ICD codes.
    diagnosis_types: list of str, optional
        Include only those diagnoses that are of certain type.
    limit: int, optional
        Limit the number of rows returned.

    """
    # Get diagnoses
    if diagnoses_table is None:
        diagnoses_table = diagnoses(**process_kwargs).query

    # Join on patient encounters
    if patient_encounters_table is None:
        patient_encounters_table = patient_encounters().query

    table = qp.Join(diagnoses_table, on=ENCOUNTER_ID)(patient_encounters_table)

    return QueryInterface(_db, table)


def room_transfers(**process_kwargs) -> QueryInterface:
    """Query room transfer data.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed table, wrapped in an interface object.

    Other Parameters
    ----------------
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(ROOM_TRANSFER)

    # Join with lookup to get transfer description.
    lookup_rt_table = get_table(LOOKUP_ROOM_TRANSFER)
    table = qp.Join(
        lookup_rt_table,
        on=("medical_service", "value"),
        join_table_cols="description",
    )(table)
    table = qp.Rename({"description": "transfer_description"})(table)

    # Process optional operations
    operations: List[tuple] = [(qp.Limit, [qp.QAP("limit")], {})]
    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
@assert_table_has_columns(patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID])
def care_units(
    patient_encounters_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query care unit data.

    Parameters
    ----------
    patient_encounters_table: cyclops.query.util.TableTypes, optional
        Patient encounters table used to join.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed table, wrapped in an interface object.

    Other Parameters
    ----------------
    limit: int, optional
        Limit the number of rows returned.

    """
    filter_care_unit_cols = qp.FilterColumns(
        [
            ENCOUNTER_ID,
            "admit",
            "discharge",
            CARE_UNIT,
        ]
    )

    # In-patient table.
    ip_table = get_table(IP_ADMIN)
    ip_table = qp.Rename(
        {
            ADMIT_TIMESTAMP: "admit",
            DISCHARGE_TIMESTAMP: "discharge",
        }
    )(ip_table)
    ip_table = qp.Literal("IP", CARE_UNIT)(ip_table)
    ip_table = filter_care_unit_cols(ip_table)

    # Special care unit table.
    scu_table = get_table(IP_SCU)
    scu_table = qp.Rename(
        {
            SCU_ADMIT_TIMESTAMP: "admit",
            SCU_DISCHARGE_TIMESTAMP: "discharge",
        }
    )(scu_table)
    scu_table = qp.Literal("SCU", CARE_UNIT)(scu_table)
    scu_table = filter_care_unit_cols(scu_table)

    # Emergency room/department table.
    er_table = er_admin().query
    er_table = qp.Rename(
        {
            ER_ADMIT_TIMESTAMP: "admit",
            ER_DISCHARGE_TIMESTAMP: "discharge",
        }
    )(er_table)
    er_table = qp.Literal("ER", CARE_UNIT)(er_table)
    er_table = filter_care_unit_cols(er_table)

    # Room transfer table.
    rt_table = room_transfers().query
    rt_table = qp.Rename(
        {
            "checkin_date_time": "admit",
            "checkout_date_time": "discharge",
        }
    )(rt_table)
    rt_table = qp.Rename({"transfer_description": CARE_UNIT})(rt_table)
    rt_table = filter_care_unit_cols(rt_table)

    # Combine.
    table = union_all(
        select(er_table),
        select(scu_table),
        select(ip_table),
        select(rt_table),
    ).subquery()

    if patient_encounters_table is not None:
        table = qp.Join(patient_encounters_table, on=ENCOUNTER_ID)(table)

    # Process optional operations
    operations: List[tuple] = [(qp.Limit, [qp.QAP("limit")], {})]
    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
@assert_table_has_columns(patient_encounters_table=[ENCOUNTER_ID, SUBJECT_ID])
def events(
    event_category: str,
    patient_encounters_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query events.

    Parameters
    ----------
    event_category : str or list of str
        Specify event category, e.g., lab, vitals, intervention, etc.
    patient_encounters_table: cyclops.query.util.TableTypes, optional
        Patient encounters table used to join.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed table, wrapped in an interface object.


    Other Parameters
    ----------------
    event_names: str or list of str, optional
        Get only certain event names.
    event_name_substring: str, optional
        Get only event names with a substring.
    limit: int, optional
        Limit the number of rows returned.

    """
    if event_category not in EVENT_CATEGORIES:
        raise ValueError(
            f"""Invalid event category specified.
            Must be in {", ".join(EVENT_CATEGORIES)}"""
        )

    table = get_table(event_category)

    # Remove events with no recorded name.
    table = qp.ConditionEquals(EVENT_NAME, EMPTY_STRING, not_=True, to_str=True)(table)

    # Process optional operations
    operations: List[tuple] = [
        (qp.ConditionIn, [EVENT_NAME, qp.QAP("event_names")], {"to_str": True}),
        (
            qp.ConditionSubstring,
            [EVENT_NAME, qp.QAP("event_name_substring")],
            {"to_str": True},
        ),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    # Join on patient encounters
    if patient_encounters_table is not None:
        table = qp.Join(patient_encounters_table, on=ENCOUNTER_ID)(table)

    return QueryInterface(_db, table)
