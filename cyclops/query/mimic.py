"""MIMIC-IV query API."""

# pylint: disable=duplicate-code

import logging
from typing import List, Optional, Union

import sqlalchemy
from sqlalchemy import Integer, extract, func, select
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops import config
from cyclops.constants import MIMIC
from cyclops.orm import Database
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DIAGNOSIS_CODE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    SEX,
    YEAR,
)
from cyclops.processors.constants import MONTH
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.postprocess.mimic import process_mimic_care_units
from cyclops.query.util import (
    DBTable,
    add_interval_to_timestamps,
    drop_attributes,
    equals,
    get_attribute,
    has_substring,
    in_,
    rename_attributes,
    reorder_attributes,
    to_datetime_format,
    to_list,
    trim_attributes,
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

_db = Database(config.read_config(MIMIC))
TABLE_MAP = {
    PATIENTS: lambda db: db.mimic_core.patients,
    ADMISSIONS: lambda db: db.mimic_core.admissions,
    DIAGNOSES: lambda db: db.mimic_hosp.d_icd_diagnoses,
    PATIENT_DIAGNOSES: lambda db: db.mimic_hosp.diagnoses_icd,
    EVENT_LABELS: lambda db: db.mimic_icu.d_items,
    EVENTS: lambda db: db.mimic_icu.chartevents,
    TRANSFERS: lambda db: db.mimic_core.transfers,
}
MIMIC_COLUMN_MAP = {
    "hadm_id": ENCOUNTER_ID,
    "admittime": ADMIT_TIMESTAMP,
    "dischtime": DISCHARGE_TIMESTAMP,
    "gender": SEX,
    "anchor_age": AGE,
    "valueuom": EVENT_VALUE_UNIT,
    "label": EVENT_NAME,
    "valuenum": EVENT_VALUE,
    "charttime": EVENT_TIMESTAMP,
    "icd_code": DIAGNOSIS_CODE,
}


def get_lookup_table(table_name: str) -> QueryInterface:
    """Get lookup table data.

    Some tables are minimal reference tables that are
    useful for reference. The entire table is wrapped as
    a query to run.

    Parameters
    ----------
    table_name: str
        Name of the table.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    if table_name not in [DIAGNOSES, EVENT_LABELS]:
        raise ValueError("Not a recognised lookup/dimension table!")
    subquery = select(TABLE_MAP[table_name](_db).data).subquery()

    return QueryInterface(_db, subquery)


def _map_table_attributes(table_name: str) -> Subquery:
    """Map queried data attributes into common column names.

    For unifying processing functions across datasets, data
    attributes are currently mapped to the same name, to allow for processing
    to recognise them.

    Parameters
    ----------
    table_name: str
        Name of MIMIC table.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Query with mapped attributes.

    """
    return rename_attributes(TABLE_MAP[table_name](_db).data, MIMIC_COLUMN_MAP)


def patients(
    years: Optional[Union[int, List[int]]] = None,
    months: Optional[Union[int, List[int]]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    process_anchor_year: bool = True,
) -> QueryInterface:
    """Query MIMIC patient data.

    Parameters
    ----------
    years: int or list of int, optional
        Years for which patient encounters are to be filtered.
    months: int or list of int, optional
        Months for which patient encounters are to be filtered.
    from_date: str, optional
        Gather patients admitted >= from_date in YYYY-MM-DD format.
    to_date: str, optional
        Gather patients admitted <= to_date in YYYY-MM-DD format.
    process_anchor_year : bool, optional
        Whether to process and include the patient's anchor
        year, i.e., to infer approximate year of care.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    patients_table = TABLE_MAP[PATIENTS](_db)
    admissions_table = TABLE_MAP[ADMISSIONS](_db)

    if not process_anchor_year and years:
        raise ValueError(
            "process_anchor_year must be set to True to use years filter correctly!"
        )
    if not process_anchor_year and from_date:
        raise ValueError("process_anchor_year not set, filtering by date won't work!")
    if not process_anchor_year and to_date:
        raise ValueError("process_anchor_year not set, filtering by date won't work!")

    if not process_anchor_year:
        return select(patients_table.data)

    # Process and include patient's anchor year.
    subquery = select(
        patients_table.data,
        (func.substr(patients_table.anchor_year_group, 1, 4).cast(Integer)).label(
            "anchor_year_group_start"
        ),
        (func.substr(patients_table.anchor_year_group, 8, 12).cast(Integer)).label(
            "anchor_year_group_end"
        ),
    ).subquery()
    subquery = select(
        subquery,
        (
            subquery.c.anchor_year_group_start
            + (subquery.c.anchor_year_group_end - subquery.c.anchor_year_group_start)
            / 2
        ).label("anchor_year_group_median"),
    ).subquery()
    subquery = select(
        subquery,
        (subquery.c.anchor_year_group_median - subquery.c.anchor_year).label(
            "anchor_year_difference"
        ),
    ).subquery()
    subquery = drop_attributes(subquery, ["anchor_year_group"])

    # Join with admissions table and add anchor year difference.
    admissions_subquery = select(admissions_table.data).subquery()
    subquery = (
        select(subquery, admissions_subquery)
        .where(subquery.c.subject_id == admissions_subquery.c.subject_id)
        .subquery()
    )
    subquery = rename_attributes(subquery, MIMIC_COLUMN_MAP)
    subquery = add_interval_to_timestamps(
        subquery,
        [ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP],
        years="anchor_year_difference",
    )

    # Filter based on years or months or specific date range (admission time).
    if years:
        subquery = (
            select(subquery)
            .where(in_(extract(YEAR, subquery.c.admit_timestamp), to_list(years)))
            .subquery()
        )
    if months:
        subquery = (
            select(subquery)
            .where(in_(extract(MONTH, subquery.c.admit_timestamp), to_list(months)))
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

    return QueryInterface(_db, subquery)


def _join_with_patients(
    patients_table: Union[Select, Subquery, Table, DBTable],
    table_: Union[Select, Subquery, Table, DBTable],
) -> QueryInterface:
    """Join a subquery with MIMIC patient static information.

    Assumes that the query has column 'encounter_id'.

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
        Patient query table.
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        A query with column encounter_id.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    if not hasattr(table_.c, ENCOUNTER_ID) or not hasattr(
        patients_table.c, ENCOUNTER_ID
    ):
        raise ValueError(
            "Input query table and patients table must have attribute 'encounter_id'."
        )

    # Join on patients (subject column).
    subquery = (
        select(table_, patients_table)
        .where(table_.c.encounter_id == patients_table.c.encounter_id)
        .subquery()
    )

    return QueryInterface(_db, subquery)


def _diagnoses(
    version: Optional[int] = None,
) -> sqlalchemy.sql.selectable.Subquery:
    """Query MIMIC possible diagnoses.

    Parameters
    ----------
    version: int, optional
        If specified, restrict ICD codes to this version.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Constructed subquery, with matched diagnoses.

    """
    # Get diagnoses.
    table_ = TABLE_MAP[DIAGNOSES](_db)
    subquery = select(table_.data).subquery()
    subquery = rename_attributes(subquery, MIMIC_COLUMN_MAP)

    # Filter by version.
    if version:
        subquery = (
            select(subquery)
            .where(equals(subquery.c.icd_version, version, to_int=True))
            .subquery()
        )

    # Trim whitespace from icd_codes.
    subquery = trim_attributes(subquery, [DIAGNOSIS_CODE])

    # Rename long_title to icd_title.
    subquery = rename_attributes(subquery, {"long_title": "icd_title"})

    # Re-order the columns nicely.
    subquery = reorder_attributes(
        subquery, [DIAGNOSIS_CODE, "icd_title", "icd_version"]
    )

    return subquery


def diagnoses(
    version: Optional[int] = None,
    substring: Optional[str] = None,
    diagnosis_codes: Union[str, List[str]] = None,
    patients: Optional[QueryInterface] = None,  # pylint: disable=redefined-outer-name
) -> QueryInterface:
    """Query MIMIC patient diagnoses.

    Parameters
    ----------
    version : int, optional
        If specified, restrict ICD codes to this version.
    substring : str
        Substring to match in an ICD code.
    diagnosis_codes : str or list of str
        The ICD codes to filter.
    patients: QueryInterface, optional
        Patient encounters query wrapped, used to join with diagnoses.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    # Get patient diagnoses.
    table_ = TABLE_MAP[PATIENT_DIAGNOSES](_db)
    subquery = select(table_.data).subquery()
    subquery = rename_attributes(subquery, MIMIC_COLUMN_MAP)

    # Filter by version
    if version:
        subquery = (
            select(subquery)
            .where(equals(subquery.c.icd_version, version, to_int=True))
            .subquery()
        )

    # Trim whitespace from icd_codes.
    subquery = trim_attributes(subquery, [DIAGNOSIS_CODE])

    # Get codes.
    code_subquery = _diagnoses(version=version)

    # Get patient diagnoses, including ICD title.
    subquery = (
        select(subquery, code_subquery.c.icd_title)
        .join(subquery, subquery.c.diagnosis_code == code_subquery.c.diagnosis_code)
        .subquery()
    )

    # Filter by substring.
    if substring:
        subquery = (
            select(subquery)
            .where(has_substring(subquery.c.icd_title, substring, to_str=True))
            .subquery()
        )

    # Filter by ICD codes.
    if diagnosis_codes:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.diagnosis_code, diagnosis_codes, to_str=True))
            .subquery()
        )

    if patients:
        return _join_with_patients(patients.query, subquery)

    return QueryInterface(_db, subquery)


def transfers(
    encounters: Optional[list] = None,
    patients: Optional[QueryInterface] = None,  # pylint: disable=redefined-outer-name
) -> QueryInterfaceProcessed:
    """Get care unit table within a given set of encounters.

    Parameters
    ----------
    encounters : list, optional
        The encounter IDs on which to filter. If None, consider all encounters.
    patients: QueryInterface, optional
        Patient encounters query wrapped, used to join with events.

    Returns
    -------
    cyclops.query.interface.QueryInterfaceProcessed
        Constructed query, wrapped in an interface object.

    """
    filter_encounters = (
        lambda query, encounters: query
        if encounters is None
        else select(query)
        .where(in_(get_attribute(query, ENCOUNTER_ID), encounters))
        .subquery()
    )

    subquery = _map_table_attributes(TRANSFERS)

    if encounters:
        subquery = filter_encounters(subquery, encounters)

    if patients:
        subquery = _join_with_patients(patients.query, subquery).query

    return QueryInterface(_db, subquery)


def care_units(
    encounters: Optional[list] = None,
    patients: Optional[QueryInterface] = None,  # pylint: disable=redefined-outer-name
) -> QueryInterfaceProcessed:
    """Get care unit table within a given set of encounters.

    Parameters
    ----------
    encounters : list, optional
        The encounter IDs on which to filter. If None, consider all encounters.
    patients: QueryInterface, optional
        Patient encounters query wrapped, used to join with events.

    Returns
    -------
    cyclops.query.interface.QueryInterfaceProcessed
        Constructed query, wrapped in an interface object.

    """
    subquery = transfers(encounters=encounters, patients=patients).query

    return QueryInterfaceProcessed(
        _db,
        subquery,
        process_fn=process_mimic_care_units,
        process_fn_kwargs={"specific": False},
    )


def events(
    category: Optional[str] = None,
    names: Optional[Union[str, List[str]]] = None,
    substring: Optional[str] = None,
    patients: Optional[QueryInterface] = None,  # pylint: disable=redefined-outer-name
) -> QueryInterface:
    """Query MIMIC events.

    Parameters
    ----------
    category : str, optional
        If specified, restrict to this category.
    names : str or list of str, optional
        The event names to filter.
    substring: str, optional
        Substring to search event names to filter.
    patients: QueryInterface, optional
        Patient encounters query wrapped, used to join with events.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[EVENTS](_db)
    event_labels_table = TABLE_MAP[EVENT_LABELS](_db)

    subquery = select(table_.data).subquery()
    subquery = rename_attributes(subquery, MIMIC_COLUMN_MAP)
    event_labels_subquery = select(event_labels_table.data).subquery()
    event_labels_subquery = rename_attributes(event_labels_subquery, MIMIC_COLUMN_MAP)

    subquery = (
        select(
            subquery,
            event_labels_subquery.c.category,
            event_labels_subquery.c.event_name,
        )
        .join(
            event_labels_subquery, subquery.c.itemid == event_labels_subquery.c.itemid
        )
        .subquery()
    )

    # Filter by category.
    if category:
        subquery = (
            select(subquery).where(equals(subquery.c.category, category)).subquery()
        )
    # Filter by event names.
    if names:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.event_name, to_list(names), to_str=True))
            .subquery()
        )
    # Filter by event substrings.
    if substring:
        subquery = (
            select(subquery)
            .where(has_substring(subquery.c.event_name, substring, to_str=True))
            .subquery()
        )

    if patients:
        subquery = _join_with_patients(patients.query, subquery).query
        subquery = add_interval_to_timestamps(
            subquery,
            [EVENT_TIMESTAMP],
            years="anchor_year_difference",
        )

    return QueryInterface(_db, subquery)
