"""MIMIC-IV query API."""

from typing import List, Union, Optional

from sqlalchemy import select, func
from sqlalchemy import Integer
from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.sql.schema import Table

from cyclops import config
from cyclops.orm import Database
from cyclops.query.utils import (
    debug_query_msg,
    query_params_to_type,
    drop_attributes,
    trim_attributes,
    rename_attributes,
    reorder_attributes,
    equals,
    has_substring,
    in_,
    DBTable,
)
from cyclops.query.interface import QueryInterface
from cyclops.constants import MIMIC


_db = Database(config.read_config(MIMIC))
TABLE_MAP = {
    "patients": lambda db: db.mimic_core.patients,
    "diagnoses": lambda db: db.mimic_hosp.d_icd_diagnoses,
    "patient_diagnoses": lambda db: db.mimic_hosp.diagnoses_icd,
    "event_labels": lambda db: db.mimic_icu.d_items,
    "events": lambda db: db.mimic_icu.chartevents,
}


@debug_query_msg
def patients(years: List[str] = None, process_anchor_year: bool = True) -> Select:
    """Query MIMIC patient data.

    Parameters
    ----------
    years: list of str, optional
        Years for which patient encounters are to be filtered.
    process_anchor_year : bool, optional
        Whether to process and include the patient's anchor
        year, i.e., year of care information.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    table_ = TABLE_MAP["patients"](_db)

    if not process_anchor_year and years:
        raise ValueError("process_anchor_year must be set to True to use years filter!")

    if not process_anchor_year:
        return select(table_.data)

    # Process and include patient's anchor year, i.e., year of care information.
    subquery = select(
        table_.data,
        (func.substr(table_.anchor_year_group, 1, 4).cast(Integer)).label(
            "anchor_year_group_start"
        ),
        (func.substr(table_.anchor_year_group, 8, 12).cast(Integer)).label(
            "anchor_year_group_end"
        ),
    ).subquery()

    subquery = select(
        subquery,
        (
            subquery.c.anchor_year_group_start
            + (subquery.c.anchor_year_group_end - subquery.c.anchor_year_group_start)
            / 2
        ).label("year"),
    ).subquery()

    subquery = select(
        subquery,
        (subquery.c.year - subquery.c.anchor_year).label("anchor_year_difference"),
    ).subquery()

    subquery = drop_attributes(subquery, ["anchor_year_group"]).subquery()

    if years:
        subquery = select(subquery).where(in_(subquery.c.year, years)).subquery()
    return QueryInterface(_db, subquery)


@debug_query_msg
@query_params_to_type(Subquery)
def join_with_patients(
    patients_table: Union[Select, Subquery, Table, DBTable],
    table_: Union[Select, Subquery, Table, DBTable],
) -> Select:
    """Join a subquery with MIMIC patient static information.

    Assumes that the query t has column 'subject_id'.

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
        Patient query table.
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        A query with column subject_id.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    if not hasattr(table_.c, "subject_id") or not hasattr(
        patients_table.c, "subject_id"
    ):
        raise ValueError(
            "Input query table and patients table must have attribute 'subject_id'."
        )

    # Join on patients (subject column).
    subquery = (
        select(table_, patients_table)
        .where(table_.c.subject_id == patients_table.c.subject_id)
        .subquery()
    )

    return QueryInterface(_db, subquery)


def _diagnoses(
    version: Optional[int] = None, substring: Optional[str] = None
) -> Select:
    """Query MIMIC possible diagnoses.

    Parameters
    ----------
    version: int, optional
        If specified, restrict ICD codes to this version.
    substring: str, optional
        Search sub-string for diagnoses, applied on the ICD long title.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Constructed subquery, with matched diagnoses.

    """
    # Get diagnoses.
    table_ = TABLE_MAP["diagnoses"](_db)
    subquery = select(table_.data).subquery()

    # Filter by version.
    if version is not None:
        subquery = (
            select(subquery)
            .where(equals(subquery.c.icd_version, version, to_int=True))
            .subquery()
        )

    # Trim whitespace from icd_codes.
    subquery = trim_attributes(subquery, ["icd_code"]).subquery()

    # Rename long_title to icd_title.
    subquery = rename_attributes(subquery, {"long_title": "icd_title"}).subquery()

    # Re-order the columns nicely.
    subquery = reorder_attributes(
        subquery, ["icd_code", "icd_title", "icd_version"]
    ).subquery()

    if substring:
        subquery = (
            select(subquery)
            .where(has_substring(subquery.c.icd_title, substring, to_str=True))
            .subquery()
        )

    return subquery


@debug_query_msg
def diagnoses(
    version: Optional[int] = None,
    substring: str = None,
    diagnosis_codes: Union[str, List[str]] = None,
) -> Select:
    """Query MIMIC patient diagnoses.

    Parameters
    ----------
    version : int, optional
        If specified, restrict ICD codes to this version.
    substring : str
        Substring to match in an ICD code.
    diagnosis_codes : str or list of str
        The ICD codes to filter.

    Returns
    -------
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    # Get patient diagnoses.
    table_ = TABLE_MAP["patient_diagnoses"](_db)
    subquery = select(table_.data).subquery()

    # Filter by version
    if version is not None:
        subquery = (
            select(subquery)
            .where(equals(subquery.c.icd_version, version, to_int=True))
            .subquery()
        )

    # Trim whitespace from icd_codes.
    query = trim_attributes(subquery, ["icd_code"])

    # Include ICD title.
    subquery = query.subquery()

    # Get codes.
    code_subquery = _diagnoses(version=version, substring=substring)

    # Get patient diagnoses, including ICD title.
    subquery = (
        select(subquery, code_subquery.c.icd_title)
        .join(subquery, subquery.c.icd_code == code_subquery.c.icd_code)
        .subquery()
    )

    # Select those in the given ICD codes.
    if diagnosis_codes:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.icd_code, diagnosis_codes, to_str=True))
            .subquery()
        )

    return QueryInterface(_db, subquery)


@debug_query_msg
def event_labels(category: Optional[str] = None) -> Select:
    """Query MIMIC event labels.

    Parameters
    ----------
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    table_ = TABLE_MAP["event_labels"](_db)
    sel = select(table_.data)

    # Filter by category.
    if category is not None:
        subquery = sel.subquery()
        return select(subquery).where(equals(subquery.c.category, category))

    return sel


@debug_query_msg
def event_labels_by_substring(substring: str, category: Optional[str] = None) -> Select:
    """Query MIMIC event labels by substring.

    Parameters
    ----------
    substring : str
        Substring to match in an event label.
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    # Get event labels.
    subquery = event_labels(category=category).subquery()

    # Get labels by label substring.
    query = select(subquery).where(has_substring(subquery.c.label, substring))

    return query


@debug_query_msg
def events(join_on_labels: bool = True, category: Optional[str] = None) -> Select:
    """Query MIMIC events.

    Parameters
    ----------
    join_on_labels : bool, default=True
        Whether to join events with event labels by itemid.
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    table_ = TABLE_MAP["events"](_db)
    sel = select(table_.data)

    # Filter by category.
    if category is not None:
        subquery = sel.subquery()
        labels_subquery = event_labels(category=category).subquery()
        subquery = (
            select(subquery, labels_subquery.c.category)
            .join(labels_subquery, subquery.c.itemid == labels_subquery.c.itemid)
            .subquery()
        )

        sel = select(subquery).where(equals(subquery.c.category, category))

    if not join_on_labels:
        return sel

    # Get and include event label.
    subquery = sel.subquery()
    query = select(
        drop_attributes(subquery, ["itemid"]).subquery(),
        _db.mimic_icu.d_items.data,
    ).filter(subquery.c.itemid == _db.mimic_icu.d_items.itemid)

    return query


@debug_query_msg
def events_by_itemids(itemids: List[int], category: Optional[str] = None) -> Select:
    """Query MIMIC events, taking only specified itemids.

    Parameters
    ----------
    itemids : list of int
        The itemids to take.
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    # Get events.
    subquery = events(category=category).subquery()

    # Get events in the itemids list.
    cond = in_(subquery.c.itemid, itemids, to_int=True)
    query = select(subquery).where(cond)

    return query


@debug_query_msg
def events_by_labels(
    labels: Union[str, List[str]], category: Optional[str] = None
) -> Select:
    """Query MIMIC events, taking only specified labels.

    Parameters
    ----------
    labels : str or list of str
        The labels to take.
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    # Get events.
    subquery = events(category=category).subquery()

    # Get those in label list.
    cond = in_(subquery.c.label, labels, lower=False, strip=False)
    query = select(subquery).where(cond)

    return query


@debug_query_msg
def events_by_label_substring(substring: str, category: Optional[str] = None) -> Select:
    """Query MIMIC events by label substring.

    Parameters
    ----------
    substring : str
        Substring to match in an event label.
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    # Get events.
    subquery = events(category=category).subquery()

    # Get by substring.
    cond = has_substring(subquery.c.label, substring)
    query = select(subquery).where(cond)

    return query
