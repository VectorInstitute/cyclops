"""MIMIC-IV queries using SQLAlchemy ORM."""

import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.sql.expression import and_
from sqlalchemy import Integer

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

from cyclops.query_utils import debug_query_msg, query_params_to_type
import cyclops.query_utils as q_utils

from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.sql.schema import Table
from cyclops.query_utils import DBTable

from typing import List, Union, Optional

# pylint: disable=singleton-comparison


TABLE = {
    "patients": lambda db: db.mimic_core.patients,
    "diagnoses": lambda db: db.mimic_hosp.d_icd_diagnoses,
    "patient_diagnoses": lambda db: db.mimic_hosp.diagnoses_icd,
    "event_labels": lambda db: db.mimic_icu.d_items,
    "events": lambda db: db.mimic_icu.chartevents,
}


@debug_query_msg
def patients(db: Database, process_anchor_year: bool = True) -> Select:
    """Query MIMIC patient data.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    process_anchor_year : bool, default=True
        Whether to process and include the patient's anchor
        year, i.e., year of care information.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    t = TABLE["patients"](db)
    
    if not process_anchor_year:
        return select(t.data)
    
    # Process and include patient's anchor year
    # i.e., year of car information
    subquery = select(
        t.data, \
        (func.substr(t.anchor_year_group, 1, 4) \
            .cast(Integer)).label("anchor_year_group_start"), \
        (func.substr(t.anchor_year_group, 8, 12) \
             .cast(Integer)).label("anchor_year_group_end")
    ).subquery()
    
    subquery = select(subquery, (subquery.c.anchor_year_group_start + \
        (subquery.c.anchor_year_group_end - \
         subquery.c.anchor_year_group_start)/2) \
         .label("year")).subquery()
    
    subquery = select(subquery, (subquery.c.year - \
        subquery.c.anchor_year).label("anchor_year_difference")).subquery()
    
    query = q_utils.drop_attributes(subquery, ["anchor_year_group"])
    return query


@debug_query_msg
@query_params_to_type(Subquery)
def join_with_patients(db: Database, t: Union[Select, Subquery, Table, DBTable], \
    process_anchor_year: bool = True) -> Select:
    """Join a subquery with MIMIC patient static information.
    Assumes that the query t has column 'subject_id'.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
        A query with column subject_id.
    process_anchor_year : bool, default=True
        Whether to process and include the patient's anchor
        year, i.e., get the year of care information.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    if not hasattr(t.c, 'subject_id'):
        raise ValueError( \
            "Subquery t must have attribute 'subject_id'.")
    
    # Get patients
    p = patients(db, \
        process_anchor_year=process_anchor_year).subquery()
    
    # Join on patients (subject column)
    query = select(t, p).where( \
        t.c.subject_id == p.c.subject_id)
    
    return query


@debug_query_msg
def diagnoses(db: Database, version: Optional[int] = None) -> Select:
    """Query MIMIC possible diagnoses.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    version : int, optional
        If specified, restrict ICD codes to this version.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get diagnoses
    t = TABLE["diagnoses"](db)
    subquery = select(t.data).subquery()
    
    # Filter by version
    if version is not None:
        subquery = select(subquery).where(q_utils.equals_cond( \
            subquery.c.icd_version, version, to_int=True)).subquery()
    
    # Trim whitespace from icd_codes
    subquery = q_utils.trim_columns(subquery, ["icd_code"]).subquery()
    
    # Rename long_title to icd_title
    subquery = q_utils.rename_attributes(subquery, \
        {"long_title":"icd_title"}).subquery()
    
    # Re-order the columns nicely
    query = q_utils.reorder_attributes(subquery, \
        ["icd_code", "icd_title", "icd_version"])
    
    return query


@debug_query_msg
def diagnoses_by_substring(db: Database, substring: str, \
    version: Optional[int] = None) -> Select:
    """Query MIMIC possible diagnoses by ICD code substring.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    substring : str
        Substring to match in an ICD code.
    version : int, optional
        If specified, restrict ICD codes to this version.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get diagnoses
    subquery = diagnoses(db, version=version).subquery()
                            
    # Get diagnoses by substring
    query = select(subquery).where( \
        q_utils.substring_cond(subquery.c.icd_title, substring))
    
    return query


@debug_query_msg
def patient_diagnoses(db: Database, version: Optional[int] = None, \
    include_icd_title: bool = True) -> Select:
    """Query MIMIC patient diagnoses.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    version : int, optional
        If specified, restrict ICD codes to this version.
    include_icd_title : bool, default=True
        Whether to include ICD code titles.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get patient diagnoses
    t = TABLE["patient_diagnoses"](db)
    subquery = select(t.data).subquery()
    
    # Filter by version
    if version is not None:
        subquery = select(subquery).where(q_utils.equals_cond( \
            subquery.c.icd_version, version, to_int=True)).subquery()
    
    # Trim whitespace from icd_codes
    query = q_utils.trim_columns(subquery, ["icd_code"])
    
    if not include_icd_title:
        return query
    
    # Include ICD title
    subquery = query.subquery()
    
    # Get codes
    code_subquery = diagnoses(db, version=version)
    
    # Get patient diagnoses, including ICD title
    query = select(subquery, code_subquery.c.icd_title) \
    .join(subquery, subquery.c.icd_code == code_subquery.c.icd_code)
    
    return query


@debug_query_msg
def patient_diagnoses_by_icd_codes(db: Database, codes: Union[str, List[str]], \
    version: Optional[int] = None) -> Select:
    """Query MIMIC patient diagnoses, taking only specified ICD codes.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    codes : str or list of str
        The ICD codes to take.
    version : int, optional
        If specified, restrict ICD codes to this version.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get patient diagnoses
    subquery = patient_diagnoses(db, version=version).subquery()
    
    # Select those in the given ICD codes
    query = select(subquery).where(q_utils.in_list_condition( \
        subquery.c.icd_code, codes, to_str=True))
    
    return query


@debug_query_msg
def patient_diagnoses_by_substring(db: Database, substring: str, \
        version: Optional[int] = None) -> Select:
    """Query MIMIC patient diagnoses by an ICD code substring.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    substring : str
        Substring to match in an ICD code.
    version : int, optional
        If specified, restrict ICD codes to this version.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get codes by substring
    code_subquery = diagnoses_by_substring(db, substring, version=version).subquery()
    
    # Get patient diagnoses
    patient_subquery = patient_diagnoses(db, version=version).subquery()
    
    # Get patient diagnoses by substring
    query = select(patient_subquery, code_subquery.c.icd_title) \
    .join(code_subquery, patient_subquery.c.icd_code == code_subquery.c.icd_code)
    
    return query


@debug_query_msg
def event_labels(db: Database, category: Optional[str] = None) -> Select:
    """Query MIMIC event labels.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    category : str, optional
        If specified, restrict to this category.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    t = TABLE["event_labels"](db)
    sel = select(t.data)
    
    # Filter by category
    if category is not None:
        subquery = sel.subquery()
        return select(subquery).where(q_utils.equals_cond( \
            subquery.c.category, category))
    
    return sel


@debug_query_msg
def event_labels_by_substring(db: Database, substring: str, \
    category: Optional[str] = None) -> Select:
    """Query MIMIC event labels by substring.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    substring : str
        Substring to match in an event label.
    category : str, optional
        If specified, restrict to this category.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get event labels
    subquery = event_labels(db, category=category).subquery()
    
    # Get labels by label substring
    query = select(subquery).where( \
        q_utils.substring_cond(subquery.c.label, substring))
    
    
    return query


@debug_query_msg
def events(db: Database, join_on_labels: bool = True, \
        category: Optional[str] = None) -> Select:
    """Query MIMIC events.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    join_on_labels : bool, default=True
        Whether to join events with event labels by itemid.
    category : str, optional
        If specified, restrict to this category.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    t = TABLE["events"](db)
    sel = select(t.data)
    
    # Filter by category
    if category is not None:
        subquery = sel.subquery()
        labels_subquery = event_labels(db, category=category).subquery()
        subquery = select(subquery, \
            labels_subquery.c.category).join(labels_subquery,
            subquery.c.itemid == labels_subquery.c.itemid).subquery()
        
        sel = select(subquery).where(q_utils.equals_cond( \
            subquery.c.category, category))
    
    if not join_on_labels:
        return sel
    
    # Get and include event label
    subquery = sel.subquery()
    query = select(q_utils.drop_attributes(subquery, \
        ['itemid']).subquery(), \
        db.mimic_icu.d_items.data).filter( \
        subquery.c.itemid == db.mimic_icu.d_items.itemid)
    
    return query


@debug_query_msg
def events_by_itemids(db: Database, itemids: List[int], \
    category: Optional[str] = None) -> Select:
    """Query MIMIC events, taking only specified itemids.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    itemids : list of int
        The itemids to take.
    category : str, optional
        If specified, restrict to this category.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get events
    subquery = events(db, category=category).subquery()
    
    # Get events in the itemids list
    cond = q_utils.in_list_condition(subquery.c.itemid, \
        itemids, to_int=True)
    query = select(subquery).where(cond)
    
    return query


@debug_query_msg
def events_by_labels(db: Database, labels: Union[str, List[str]], \
    category: Optional[str] = None) -> Select:
    """Query MIMIC events, taking only specified labels.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    labels : str or list of str
        The labels to take.
    category : str, optional
        If specified, restrict to this category.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get events
    subquery = events(db, category=category).subquery()
    
    # Get those in label list
    cond = q_utils.in_list_condition(subquery.c.label, \
        labels, lower=False, strip=False)
    query = select(subquery).where(cond)
    
    return query


@debug_query_msg
def events_by_label_substring(db: Database, substring: str, \
    category: Optional[str] = None) -> Select:
    """Query MIMIC events by label substring.
    
    Parameters
    ----------
    db : cyclops.orm.Database
        Database ORM object.
    substring : str
        Substring to match in an event label.
    category : str, optional
        If specified, restrict to this category.
    
    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.
    """
    # Get events
    subquery = events(db, category=category).subquery()
    
    # Get by substring
    cond = q_utils.substring_cond(subquery.c.label, substring)
    query = select(subquery).where(cond)
    
    return query