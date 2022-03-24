"""GEMINI queries using SQLAlchemy ORM towards different models."""

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

# pylint: disable=singleton-comparison


@debug_query_msg
def query_gemini_delirium_diagnosis(database: Database) -> pd.DataFrame:
    """Query lab data for delirium subset.

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
            database.public.diagnosis.diagnosis_code.label(DIAGNOSIS_CODE),
            database.public.diagnosis.diagnosis_type,
            database.public.diagnosis.is_er_diagnosis,
        )
        .join(
            database.public.diagnosis.data,
            database.public.ip_administrative.genc_id
            == database.public.diagnosis.genc_id,
        )
        .where(database.public.ip_administrative.gemini_cohort == True)  # noqa: E712
    )
    return database.run_query(query)


@debug_query_msg
def query_gemini_delirium_lab(database: Database) -> pd.DataFrame:
    """Query lab data for delirium subset.

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
def query_gemini_admin_vitals(
    database: Database, years: list, hospitals: list
) -> pd.DataFrame:
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


if __name__ == "__main__":
    cfg = config.read_config("../configs/default/*.yaml")
    db = Database(cfg)
    data = query_gemini_delirium_lab(db)
    print(len(data))
    data.to_hdf(
        "/mnt/nfs/project/delirium/_extract/delirium_extract.h5",
        key="query_gemini_delirium_lab",
    )
    data = query_gemini_delirium_diagnosis(db)
    print(len(data))
    data.to_hdf(
        "/mnt/nfs/project/delirium/_extract/delirium_extract.h5",
        key="query_gemini_delirium_diagnosis",
    )
    data = query_gemini_admin_vitals(db, [2020], [SMH])
    print(len(data))
    data.to_hdf(
        "/mnt/nfs/project/delirium/_extract/extract.h5",
        key="query_gemini_admin_vitals",
    )
