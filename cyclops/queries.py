"""Queries using SQLAlchemy ORM towards different models."""

import logging
from typing import Callable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.sql.expression import and_

import config
from cyclops.orm import Database
from cyclops.processors.constants import EMPTY_STRING
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
            db.public.ip_administrative.genc_id,
            db.public.ip_administrative.hospital_id,
            db.public.ip_administrative.admit_date_time,
            db.public.ip_administrative.discharge_date_time,
            db.public.ip_administrative.del_present,
            db.public.ip_administrative.gemini_cohort,
            db.public.lab.lab_test_name_mapped,
            db.public.lab.result_value,
            db.public.lab.result_unit,
            db.public.lab.sample_collection_date_time,
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


if __name__ == "__main__":
    cfg = config.read_config("../configs/default/*.yaml")
    db = Database(cfg)
    data = query_gemini_delirium_lab(db)
    print(len(data))
