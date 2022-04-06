"""GEMINI query API."""

from typing import List, Optional, Union

from sqlalchemy import extract, select
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops import config
from cyclops.constants import GEMINI
from cyclops.orm import Database
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    SEX,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_TIMESTAMP,
    VITAL_MEASUREMENT_VALUE,
)
from cyclops.processors.constants import EMPTY_STRING, YEAR
from cyclops.query.interface import QueryInterface
from cyclops.query.utils import (
    DBTable,
    debug_query_msg,
    equals,
    has_substring,
    in_,
    not_equals,
    query_params_to_type,
    rename_attributes,
    to_datetime_format,
)

IP_ADMIN = "ip_admin"
ER_ADMIN = "er_admin"
DIAGNOSIS = "diagnosis"
LAB = "lab"
VITALS = "vitals"
PHARMACY = "pharmacy"
INTERVENTION = "intervention"


_db = Database(config.read_config(GEMINI))
TABLE_MAP = {
    IP_ADMIN: lambda db: db.public.ip_administrative,
    ER_ADMIN: lambda db: db.public.er_administrative,
    DIAGNOSIS: lambda db: db.public.diagnosis,
    LAB: lambda db: db.public.lab,
    VITALS: lambda db: db.public.vitals,
    PHARMACY: lambda db: db.public.pharmacy,
    INTERVENTION: lambda db: db.public.intervention,
}
GEMINI_COLUMN_MAP = {
    "genc_id": ENCOUNTER_ID,
    "admit_date_time": ADMIT_TIMESTAMP,
    "discharge_date_time": DISCHARGE_TIMESTAMP,
    "gender": SEX,
    "result_value": LAB_TEST_RESULT_VALUE,
    "result_unit": LAB_TEST_RESULT_UNIT,
    "lab_test_name_mapped": LAB_TEST_NAME,
    "sample_collection_date_time": LAB_TEST_TIMESTAMP,
    "measurement_mapped": VITAL_MEASUREMENT_NAME,
    "measurement_value": VITAL_MEASUREMENT_VALUE,
    "measure_date_time": VITAL_MEASUREMENT_TIMESTAMP,
}


@debug_query_msg
def patients(
    years: List[str] = None,
    hospitals: List[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    delirium_cohort: Optional[bool] = False,
) -> QueryInterface:
    """Query patient encounters.

    Parameters
    ----------
    years: list of str, optional
        Years for which patient encounters are to be filtered.
    hospitals: list of str, optional
        Hospital sites from which patient encounters are to be filtered.
    from_date: str, optional
        Gather patients admitted >= from_date in YYYY-MM-DD format.
    to_date: str, optional
        Gather patients admitted <= to_date in YYYY-MM-DD format.
    delirium_cohort: bool, optional
        Gather patient encounters for which delirium label is available.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[IP_ADMIN](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()

    if years:
        subquery = (
            select(subquery)
            .where(in_(extract(YEAR, subquery.c.admit_timestamp), years))
            .subquery()
        )
    if hospitals:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.hospital_id, hospitals, to_str=True))
            .subquery()
        )
    if from_date:
        subquery = (
            select(subquery)
            .where(subquery.c.admit_timestamp >= to_datetime_format(from_date))
            .subquery()
        )
    if to_date:
        subquery = (
            select(subquery)
            .where(subquery.c.admit_timestamp <= to_datetime_format(to_date))
            .subquery()
        )
    if delirium_cohort:
        subquery = (
            select(subquery).where(equals(subquery.c.gemini_cohort, True)).subquery()
        )

    return QueryInterface(_db, subquery)


@query_params_to_type(Subquery)
def _join_with_patients(
    patients_table: Union[Select, Subquery, Table, DBTable],
    table_: Union[Select, Subquery, Table, DBTable],
) -> QueryInterface:
    """Join a subquery with GEMINI patient static information.

    Assumes that both query tables have "encounter_id".

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
        Patient query table.
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        A query table such as labs or vitals.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    if not hasattr(table_.c, ENCOUNTER_ID) or not hasattr(
        patients_table.c, ENCOUNTER_ID
    ):
        raise ValueError("Input query table and patients table must have encounter_id!")

    # Join on patients (encounter_id).
    query = select(table_, patients_table).where(
        table_.c.encounter_id == patients_table.c.encounter_id
    )
    return QueryInterface(_db, query)


@debug_query_msg
def diagnoses(
    diagnosis_codes: Union[List[str], str] = None, diagnosis_types: List[str] = None, patients: Optional[QueryInterface] = None
) -> QueryInterface:
    """Query diagnosis data.

    Parameters
    ----------
    diagnosis_codes: list of str or str, optional
        Names of diagnosis codes to include, or a diagnosis code search string,
        all diagnosis data are included if not provided.
    diagnosis_types: list of str, optional
        Include only those diagnoses that are of certain type.

    The following types of diagnoses are available:
    M         Most Responsible Diagnosis
    1              Pre-Admit Comorbidity
    2             Post-Admit Comorbidity
    3                Secondary Diagnosis
    4                   Morphology Codes
    5                Admitting Diagnosis
    6   Proxy Most Responsible Diagnosis
    9      External Cause of Injury Code
    0                            Newborn
    W   First Service Transfer Diagnosis
    X  Second Service Transfer Diagnosis
    Y   Third Service Transfer Diagnosis

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[DIAGNOSIS](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()
    if diagnosis_codes:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.diagnosis_code, diagnosis_codes, to_str=True))
            .subquery()
        )
    if diagnosis_types:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.diagnosis_type, diagnosis_types, to_str=True))
            .subquery()
        )
    if patients:
        return _join_with_patients(patients.query, subquery)
    
    return QueryInterface(_db, subquery)


@debug_query_msg
def labs(lab_tests: Union[List[str], str] = None, patients: Optional[QueryInterface] = None) -> QueryInterface:
    """Query lab data.

    Parameters
    ----------
    lab_tests: list of str, optional
        Names of lab tests to include, or a lab test name search string,
        all lab tests are included if not provided.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[LAB](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()
    subquery = (
        select(subquery)
        .where(not_equals(subquery.c.lab_test_name, EMPTY_STRING, to_str=True))
        .subquery()
    )
    if lab_tests and isinstance(lab_tests, list):
        subquery = (
            select(subquery)
            .where(in_(subquery.c.lab_test_name, lab_tests, to_str=True))
            .subquery()
        )
    elif lab_tests and isinstance(lab_tests, str):
        subquery = (
            select(subquery)
            .where(has_substring(subquery.c.lab_test_name, lab_tests, to_str=True))
            .subquery()
        )
    if patients:
        return _join_with_patients(patients.query, subquery)
    
    return QueryInterface(_db, subquery)


@debug_query_msg
def vitals(vital_names: Union[List[str], str] = None, patients: Optional[QueryInterface] = None) -> QueryInterface:
    """Query vitals data.

    Parameters
    ----------
    vital_names: list of str or str, optional
        Names of vital measurements to include, or a vital name search string,
        all measurements are included if not provided.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[VITALS](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()
    subquery = (
        select(subquery)
        .where(not_equals(subquery.c.vital_measurement_name, EMPTY_STRING, to_str=True))
        .subquery()
    )
    if vital_names and isinstance(vital_names, list):
        subquery = (
            select(subquery)
            .where(in_(subquery.c.vital_measurement_name, vital_names, to_str=True))
            .subquery()
        )
    if vital_names and isinstance(vital_names, str):
        subquery = (
            select(subquery)
            .where(
                has_substring(
                    subquery.c.vital_measurement_name, vital_names, to_str=True
                )
            )
            .subquery()
        )
    if patients:
        return _join_with_patients(patients.query, subquery)
    
    return QueryInterface(_db, subquery)
