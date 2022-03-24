"""MIMIC-IV queries using SQLAlchemy ORM."""

from typing import List, Union, Optional

from sqlalchemy import select, func
from sqlalchemy import Integer
from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.sql.schema import Table

from cyclops.orm import Database
import cyclops.queries.utils as q_utils
from cyclops.queries.utils import debug_query_msg, query_params_to_type
from cyclops.queries.utils import DBTable


TABLE = {
    "patients": lambda db: db.mimic_core.patients,
    "diagnoses": lambda db: db.mimic_hosp.d_icd_diagnoses,
    "patient_diagnoses": lambda db: db.mimic_hosp.diagnoses_icd,
    "event_labels": lambda db: db.mimic_icu.d_items,
    "events": lambda db: db.mimic_icu.chartevents,
}


@debug_query_msg
def patients(database: Database, process_anchor_year: bool = True) -> Select:
    """Query MIMIC patient data.

    Parameters
    ----------
    database : cyclops.orm.Database
        Database ORM object.
    process_anchor_year : bool, default=True
        Whether to process and include the patient's anchor
        year, i.e., year of care information.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    table_ = TABLE["patients"](database)

    if not process_anchor_year:
        return select(table_.data)

    # Process and include patient's anchor year
    # i.e., year of car information
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

    query = q_utils.drop_attributes(subquery, ["anchor_year_group"])
    return query


@debug_query_msg
@query_params_to_type(Subquery)
def join_with_patients(
    database: Database,
    table_: Union[Select, Subquery, Table, DBTable],
    process_anchor_year: bool = True,
) -> Select:
    """Join a subquery with MIMIC patient static information.

    Assumes that the query t has column 'subject_id'.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database ORM object.
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        A query with column subject_id.
    process_anchor_year : bool, default=True
        Whether to process and include the patient's anchor
        year, i.e., get the year of care information.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    if not hasattr(table_.c, "subject_id"):
        raise ValueError("Subquery t must have attribute 'subject_id'.")

    # Get patients
    patients_ = patients(database, process_anchor_year=process_anchor_year).subquery()

    # Join on patients (subject column)
    query = select(table_, patients_).where(
        table_.c.subject_id == patients_.c.subject_id
    )

    return query


@debug_query_msg
def diagnoses(database: Database, version: Optional[int] = None) -> Select:
    """Query MIMIC possible diagnoses.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database ORM object.
    version : int, optional
        If specified, restrict ICD codes to this version.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    # Get diagnoses
    table_ = TABLE["diagnoses"](database)
    subquery = select(table_.data).subquery()

    # Filter by version
    if version is not None:
        subquery = (
            select(subquery)
            .where(q_utils.equals_cond(subquery.c.icd_version, version, to_int=True))
            .subquery()
        )

    # Trim whitespace from icd_codes
    subquery = q_utils.trim_attributes(subquery, ["icd_code"]).subquery()

    # Rename long_title to icd_title
    subquery = q_utils.rename_attributes(
        subquery, {"long_title": "icd_title"}
    ).subquery()

    # Re-order the columns nicely
    query = q_utils.reorder_attributes(
        subquery, ["icd_code", "icd_title", "icd_version"]
    )

    return query


@debug_query_msg
def diagnoses_by_substring(
    database: Database, substring: str, version: Optional[int] = None
) -> Select:
    """Query MIMIC possible diagnoses by ICD code substring.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    subquery = diagnoses(database, version=version).subquery()

    # Get diagnoses by substring
    query = select(subquery).where(
        q_utils.substring_cond(subquery.c.icd_title, substring)
    )

    return query


@debug_query_msg
def patient_diagnoses(
    database: Database, version: Optional[int] = None, include_icd_title: bool = True
) -> Select:
    """Query MIMIC patient diagnoses.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    table_ = TABLE["patient_diagnoses"](database)
    subquery = select(table_.data).subquery()

    # Filter by version
    if version is not None:
        subquery = (
            select(subquery)
            .where(q_utils.equals_cond(subquery.c.icd_version, version, to_int=True))
            .subquery()
        )

    # Trim whitespace from icd_codes
    query = q_utils.trim_attributes(subquery, ["icd_code"])

    if not include_icd_title:
        return query

    # Include ICD title
    subquery = query.subquery()

    # Get codes
    code_subquery = diagnoses(database, version=version)

    # Get patient diagnoses, including ICD title
    query = select(subquery, code_subquery.c.icd_title).join(
        subquery, subquery.c.icd_code == code_subquery.c.icd_code
    )

    return query


@debug_query_msg
def patient_diagnoses_by_icd_codes(
    database: Database, codes: Union[str, List[str]], version: Optional[int] = None
) -> Select:
    """Query MIMIC patient diagnoses, taking only specified ICD codes.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    subquery = patient_diagnoses(database, version=version).subquery()

    # Select those in the given ICD codes
    query = select(subquery).where(
        q_utils.in_list_condition(subquery.c.icd_code, codes, to_str=True)
    )

    return query


@debug_query_msg
def patient_diagnoses_by_substring(
    database: Database, substring: str, version: Optional[int] = None
) -> Select:
    """Query MIMIC patient diagnoses by an ICD code substring.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    code_subquery = diagnoses_by_substring(
        database, substring, version=version
    ).subquery()

    # Get patient diagnoses
    patient_subquery = patient_diagnoses(database, version=version).subquery()

    # Get patient diagnoses by substring
    query = select(patient_subquery, code_subquery.c.icd_title).join(
        code_subquery, patient_subquery.c.icd_code == code_subquery.c.icd_code
    )

    return query


@debug_query_msg
def event_labels(database: Database, category: Optional[str] = None) -> Select:
    """Query MIMIC event labels.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database ORM object.
    category : str, optional
        If specified, restrict to this category.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Select ORM object.

    """
    table_ = TABLE["event_labels"](database)
    sel = select(table_.data)

    # Filter by category
    if category is not None:
        subquery = sel.subquery()
        return select(subquery).where(
            q_utils.equals_cond(subquery.c.category, category)
        )

    return sel


@debug_query_msg
def event_labels_by_substring(
    database: Database, substring: str, category: Optional[str] = None
) -> Select:
    """Query MIMIC event labels by substring.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    subquery = event_labels(database, category=category).subquery()

    # Get labels by label substring
    query = select(subquery).where(q_utils.substring_cond(subquery.c.label, substring))

    return query


@debug_query_msg
def events(
    database: Database, join_on_labels: bool = True, category: Optional[str] = None
) -> Select:
    """Query MIMIC events.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    table_ = TABLE["events"](database)
    sel = select(table_.data)

    # Filter by category
    if category is not None:
        subquery = sel.subquery()
        labels_subquery = event_labels(database, category=category).subquery()
        subquery = (
            select(subquery, labels_subquery.c.category)
            .join(labels_subquery, subquery.c.itemid == labels_subquery.c.itemid)
            .subquery()
        )

        sel = select(subquery).where(q_utils.equals_cond(subquery.c.category, category))

    if not join_on_labels:
        return sel

    # Get and include event label
    subquery = sel.subquery()
    query = select(
        q_utils.drop_attributes(subquery, ["itemid"]).subquery(),
        database.mimic_icu.d_items.data,
    ).filter(subquery.c.itemid == database.mimic_icu.d_items.itemid)

    return query


@debug_query_msg
def events_by_itemids(
    database: Database, itemids: List[int], category: Optional[str] = None
) -> Select:
    """Query MIMIC events, taking only specified itemids.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    subquery = events(database, category=category).subquery()

    # Get events in the itemids list
    cond = q_utils.in_list_condition(subquery.c.itemid, itemids, to_int=True)
    query = select(subquery).where(cond)

    return query


@debug_query_msg
def events_by_labels(
    database: Database, labels: Union[str, List[str]], category: Optional[str] = None
) -> Select:
    """Query MIMIC events, taking only specified labels.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    subquery = events(database, category=category).subquery()

    # Get those in label list
    cond = q_utils.in_list_condition(subquery.c.label, labels, lower=False, strip=False)
    query = select(subquery).where(cond)

    return query


@debug_query_msg
def events_by_label_substring(
    database: Database, substring: str, category: Optional[str] = None
) -> Select:
    """Query MIMIC events by label substring.

    Parameters
    ----------
    database: cyclops.orm.Database
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
    subquery = events(database, category=category).subquery()

    # Get by substring
    cond = q_utils.substring_cond(subquery.c.label, substring)
    query = select(subquery).where(cond)

    return query
