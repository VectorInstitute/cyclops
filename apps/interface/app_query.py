"""Querying functions used in the query page."""

from typing import Dict

import pandas as pd
from sqlalchemy import and_, select

import cyclops.query.mimiciv as db
from cyclops.processors.column_names import AGE, ENCOUNTER_ID, SUBJECT_ID
from cyclops.query.util import get_column

from .consts import APP_DIAG, APP_ENC  # , APP_EVENT


def patient_encounters(kwargs, age_min, age_max):
    """Query patient encounters."""
    table = db.patient_encounters(**kwargs).query

    # Handle age
    cond = None
    age_col = get_column(table, AGE)
    if age_min is not None and age_max is not None:
        cond = and_(age_col >= age_min, age_col <= age_max)
    elif age_min is not None:
        cond = age_col >= age_min
    elif age_max is not None:
        cond = age_col <= age_max

    if cond is not None:
        table = select(table).where(cond)

    # Return interface
    return db.get_interface(table)


def patient_diagnoses(kwargs):
    """Query patient diagnoses."""
    return db.patient_diagnoses(**kwargs)


def query(  # pylint: disable=too-many-arguments
    encounter_checked,
    encounter_kwargs,
    age_min,
    age_max,
    diagnosis_checked,
    diagnosis_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Query the relevant data from the interface."""
    datas = {}

    encounters = None

    if encounter_checked:
        encounters = patient_encounters(encounter_kwargs, age_min, age_max).run()
        datas[APP_ENC] = encounters

    if diagnosis_checked:
        if encounters is None:
            diagnoses = db.diagnoses(**diagnosis_kwargs).run()
            datas[APP_DIAG] = diagnoses
        else:
            diagnoses = patient_diagnoses(diagnosis_kwargs).run()
            datas[APP_ENC] = pd.merge(
                encounters, diagnoses.drop(SUBJECT_ID, axis=1), on=ENCOUNTER_ID
            )

    # datas[APP_EVENT] = ...

    return datas
