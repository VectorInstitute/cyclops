"""GEMINI queries using SQLAlchemy ORM towards different models."""

from typing import List

import pandas as pd
from sqlalchemy import select, extract
from sqlalchemy.sql.expression import and_

import config

from cyclops.orm import Database
from cyclops.processors.constants import EMPTY_STRING, SMH, YEAR
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    AGE,
    SEX,
    HOSPITAL_ID,
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    DISCHARGE_DISPOSITION,
    READMISSION,
    DIAGNOSIS_CODE,
    LAB_TEST_RESULT_VALUE,
    LAB_TEST_TIMESTAMP,
    LAB_TEST_NAME,
    LAB_TEST_RESULT_UNIT,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_VALUE,
    VITAL_MEASUREMENT_TIMESTAMP,
    REFERENCE_RANGE,
)
from cyclops.queries.utils import debug_query_msg
from cyclops.constants import GEMINI


IP_ADMIN = "ip_admin"
ER_ADMIN = "er_admin"
DIAGNOSIS = "diagnosis"
LAB = "lab"
VITALS = "vitals"
PHARMACY = "pharmacy"
INTERVENTION = "intervention"


_db = Database(config.read_config(GEMINI))
TABLE_MAP = {
    IP_ADMIN: _db.public.ip_administrative,
    ER_ADMIN: _db.public.er_administrative,
    DIAGNOSIS: _db.public.diagnosis,
    LAB: _db.public.lab,
    VITALS: _db.public.vitals,
    PHARMACY: _db.public.pharmacy,
    INTERVENTION: _db.public.intervention,
}


@debug_query_msg
def patients(
    year: str = None,
    hospitals: List[str] = None,
    from_date: str = None,
    to_date: str = None,
    delirium_cohort: bool = False,
) -> Select:
    """Query patient encounters.

    Parameters
    ----------
    year: str
        Gather patient encounters only from specified year.
    hospitals: list
        Gather patient encounters from specified sites.
    from_date: str, optional
        Gather patients admitted >= from_date.
    to_date: str, optional
        Gather patients admitted <= to_date.
    delirium_cohort: bool, optional
        Gather patient encounters for which delirium label is available.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    table_ = TABLE_MAP[IP_ADMIN]
    subquery = select(table_.data).subquery()

    return query


@debug_query_msg
def labs(database: Database) -> pd.DataFrame:
    """Query lab data.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database ORM object.

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.

    """
    query = (
        select(
            database.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            database.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            database.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            database.public.ip_administrative.discharge_date_time.label(
                DISCHARGE_TIMESTAMP
            ),
            database.public.ip_administrative.del_present,
            database.public.ip_administrative.gemini_cohort,
            database.public.lab.lab_test_name_mapped.label(LAB_TEST_NAME),
            database.public.lab.result_value.label(LAB_TEST_RESULT_VALUE),
            database.public.lab.result_unit.label(LAB_TEST_RESULT_UNIT),
            database.public.lab.sample_collection_date_time.label(LAB_TEST_TIMESTAMP),
            database.public.lab.reference_range.label(REFERENCE_RANGE),
        )
        .join(
            database.public.lab.data,
            database.public.ip_administrative.genc_id == database.public.lab.genc_id,
        )
        .where(
            and_(
                database.public.ip_administrative.gemini_cohort == True,  # noqa: E712
                database.public.lab.lab_test_name_mapped != EMPTY_STRING,
            )
        )
    )
    return database.run_query(query)


@debug_query_msg
def vitals(database: Database, years: list, hospitals: list) -> pd.DataFrame:
    """Query admin + vitals data filtering by hospitals and years.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database ORM object.
    years: list
        Specific year(s) to filter vitals, e.g. [2019, 2020].
    hospitals: str
        Specific hospital site(s) to apply as filter e.g. ['SMH'].

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.

    """
    query = (
        select(
            database.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            database.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            database.public.ip_administrative.age.label(AGE),
            database.public.ip_administrative.gender.label(SEX),
            database.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            database.public.ip_administrative.discharge_date_time.label(
                DISCHARGE_TIMESTAMP
            ),
            database.public.ip_administrative.discharge_disposition.label(
                DISCHARGE_DISPOSITION
            ),
            database.public.ip_administrative.readmission.label(READMISSION),
            database.public.vitals.measurement_mapped.label(VITAL_MEASUREMENT_NAME),
            database.public.vitals.measurement_value.label(VITAL_MEASUREMENT_VALUE),
            database.public.vitals.measure_date_time.label(VITAL_MEASUREMENT_TIMESTAMP),
            database.public.vitals.reference_range.label(REFERENCE_RANGE),
        )
        .where(
            and_(
                database.public.ip_administrative.hospital_id.in_(hospitals),
                extract(YEAR, database.public.ip_administrative.admit_date_time).in_(
                    years
                ),
            )
        )
        .join(
            database.public.vitals.data,
            database.public.ip_administrative.genc_id == database.public.vitals.genc_id,
        )
        .where(
            database.public.vitals.measurement_mapped != EMPTY_STRING,
        )
    )
    return database.run_query(query)
