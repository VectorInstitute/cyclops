"""Queries using SQLAlchemy ORM towards different models."""

import pandas as pd
from sqlalchemy import select, extract
from sqlalchemy.sql.expression import and_

import config
from codebase_ops import get_log_file_path

from cyclops.orm import Database
from cyclops.processors.constants import EMPTY_STRING, SMH, YEAR
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    HOSPITAL_ID,
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
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

from cyclops.query_utils import debug_query_msg
import cyclops.query_utils as q_utils

# pylint: disable=singleton-comparison


@debug_query_msg
def query_gemini_delirium_diagnosis(gemini: Database) -> pd.DataFrame:
    """Query lab data for delirium subset.

    Parameters
    ----------
    gemini: cyclops.orm.Database
        Database ORM object.

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.
    """
    query = (
        select(
            gemini.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            gemini.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            gemini.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            gemini.public.ip_administrative.discharge_date_time.label(
                DISCHARGE_TIMESTAMP
            ),
            gemini.public.ip_administrative.del_present,
            gemini.public.ip_administrative.gemini_cohort,
            gemini.public.diagnosis.diagnosis_code.label(DIAGNOSIS_CODE),
            gemini.public.diagnosis.diagnosis_type,
            gemini.public.diagnosis.is_er_diagnosis,
        )
        .join(
            gemini.public.diagnosis.data,
            gemini.public.ip_administrative.genc_id == gemini.public.diagnosis.genc_id,
        )
        .where(gemini.public.ip_administrative.gemini_cohort == True)  # noqa: E712
    )
    return gemini.run_query(query)


@debug_query_msg
def query_gemini_delirium_lab(gemini: Database) -> pd.DataFrame:
    """Query lab data for delirium subset.

    Parameters
    ----------
    gemini: cyclops.orm.Database
        Database ORM object.

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.
    """
    query = (
        select(
            gemini.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            gemini.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            gemini.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            gemini.public.ip_administrative.discharge_date_time.label(
                DISCHARGE_TIMESTAMP
            ),
            gemini.public.ip_administrative.del_present,
            gemini.public.ip_administrative.gemini_cohort,
            gemini.public.lab.lab_test_name_mapped.label(LAB_TEST_NAME),
            gemini.public.lab.result_value.label(LAB_TEST_RESULT_VALUE),
            gemini.public.lab.result_unit.label(LAB_TEST_RESULT_UNIT),
            gemini.public.lab.sample_collection_date_time.label(LAB_TEST_TIMESTAMP),
            gemini.public.lab.reference_range.label(REFERENCE_RANGE),
        )
        .join(
            gemini.public.lab.data,
            gemini.public.ip_administrative.genc_id == gemini.public.lab.genc_id,
        )
        .where(
            and_(
                gemini.public.ip_administrative.gemini_cohort == True,  # noqa: E712
                gemini.public.lab.lab_test_name_mapped != EMPTY_STRING,
            )
        )
    )
    return gemini.run_query(query)


@debug_query_msg
def query_gemini_vitals(gemini: Database, year: int, hospital: str) -> pd.DataFrame:
    """Query vitals data for delirium subset.

    Parameters
    ----------
    gemini: cyclops.orm.Database
        Database ORM object.
    year: int
        Specific year to filter vitals.
    hospital: str
        Hospital site to apply as filter.

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.
    """
    query = (
        select(
            gemini.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            gemini.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            gemini.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            gemini.public.ip_administrative.discharge_date_time.label(
                DISCHARGE_TIMESTAMP
            ),
            gemini.public.vitals.measurement_mapped.label(VITAL_MEASUREMENT_NAME),
            gemini.public.vitals.measurement_value.label(VITAL_MEASUREMENT_VALUE),
            gemini.public.vitals.measure_date_time.label(VITAL_MEASUREMENT_TIMESTAMP),
            gemini.public.vitals.reference_range.label(REFERENCE_RANGE),
        )
        .where(
            and_(
                gemini.public.ip_administrative.hospital_id == hospital,
                extract(YEAR, gemini.public.ip_administrative.admit_date_time) == year,
            )
        )
        .join(
            gemini.public.vitals.data,
            gemini.public.ip_administrative.genc_id == db.public.vitals.genc_id,
        )
        .where(
            gemini.public.vitals.measurement_mapped != EMPTY_STRING,
        )
    )
    return gemini.run_query(query)


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
    data = query_gemini_vitals(db, 2020, SMH)
    print(len(data))
    data.to_hdf(
        "/mnt/nfs/project/delirium/_extract/extract.h5",
        key="query_gemini_vitals",
    )
