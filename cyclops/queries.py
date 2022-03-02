"""Queries using SQLAlchemy ORM towards different models."""

import logging
from typing import Callable

import pandas as pd
from sqlalchemy import select, extract
from sqlalchemy.sql.expression import and_

import config
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
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def debug_query_msg(func: Callable) -> Callable:
    """Debug message decorator function.

    Parameters
    ----------
    func: function
        Function to apply decorator.

    Returns
    -------
    Callable
        Wrapper function to apply as decorator.
    """

    def wrapper_func(*args, **kwargs):
        LOGGER.debug(f"Running query function: {func.__name__}")
        query_result = func(*args, **kwargs)
        LOGGER.debug(f"Finished query function: {func.__name__}")
        return query_result

    return wrapper_func


@debug_query_msg
def query_gemini_delirium_diagnosis(db: Database) -> pd.DataFrame:
    """Query lab data for delirium subset.

    Parameters
    ----------
    db: cyclops.orm.Database
        Database ORM object.

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.
    """
    query = (
        select(
            db.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            db.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            db.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            db.public.ip_administrative.discharge_date_time.label(DISCHARGE_TIMESTAMP),
            db.public.ip_administrative.del_present,
            db.public.ip_administrative.gemini_cohort,
            db.public.diagnosis.diagnosis_code.label(DIAGNOSIS_CODE),
            db.public.diagnosis.diagnosis_type,
            db.public.diagnosis.is_er_diagnosis,
        )
        .join(
            db.public.diagnosis.x,
            db.public.ip_administrative.genc_id == db.public.diagnosis.genc_id,
        )
        .where(db.public.ip_administrative.gemini_cohort == True)  # noqa: E712
    )
    return db.run_query(query)


@debug_query_msg
def query_gemini_delirium_lab(db: Database) -> pd.DataFrame:
    """Query lab data for delirium subset.

    Parameters
    ----------
    db: cyclops.orm.Database
        Database ORM object.

    Returns
    -------
    pandas.DataFrame
        Extracted data from query.
    """
    query = (
        select(
            db.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            db.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            db.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            db.public.ip_administrative.discharge_date_time.label(DISCHARGE_TIMESTAMP),
            db.public.ip_administrative.del_present,
            db.public.ip_administrative.gemini_cohort,
            db.public.lab.lab_test_name_mapped.label(LAB_TEST_NAME),
            db.public.lab.result_value.label(LAB_TEST_RESULT_VALUE),
            db.public.lab.result_unit.label(LAB_TEST_RESULT_UNIT),
            db.public.lab.sample_collection_date_time.label(LAB_TEST_TIMESTAMP),
            db.public.lab.reference_range.label(REFERENCE_RANGE),
        )
        .join(
            db.public.lab.x,
            db.public.ip_administrative.genc_id == db.public.lab.genc_id,
        )
        .where(
            and_(
                db.public.ip_administrative.gemini_cohort == True,  # noqa: E712
                db.public.lab.lab_test_name_mapped != EMPTY_STRING,
            )
        )
    )
    return db.run_query(query)


@debug_query_msg
def query_gemini_vitals(db: Database, year: int, hospital: str) -> pd.DataFrame:
    """Query vitals data for delirium subset.

    Parameters
    ----------
    db: cyclops.orm.Database
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
            db.public.ip_administrative.genc_id.label(ENCOUNTER_ID),
            db.public.ip_administrative.hospital_id.label(HOSPITAL_ID),
            db.public.ip_administrative.admit_date_time.label(ADMIT_TIMESTAMP),
            db.public.ip_administrative.discharge_date_time.label(DISCHARGE_TIMESTAMP),
            db.public.vitals.measurement_mapped.label(VITAL_MEASUREMENT_NAME),
            db.public.vitals.measurement_value.label(VITAL_MEASUREMENT_VALUE),
            db.public.vitals.measure_date_time.label(VITAL_MEASUREMENT_TIMESTAMP),
            db.public.vitals.reference_range.label(REFERENCE_RANGE),
        )
        .where(
            and_(
                db.public.ip_administrative.hospital_id == hospital,
                extract(YEAR, db.public.ip_administrative.admit_date_time) == year,
            )
        )
        .join(
            db.public.vitals.x,
            db.public.ip_administrative.genc_id == db.public.vitals.genc_id,
        )
        .where(
            db.public.vitals.measurement_mapped != EMPTY_STRING,
        )
    )
    return db.run_query(query)


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
