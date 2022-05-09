"""GEMINI query API."""

from typing import List, Optional, Union

from sqlalchemy import extract, select
from sqlalchemy.sql.expression import literal, union_all
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops import config
from cyclops.constants import GEMINI
from cyclops.orm import Database
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    ER_ADMIT_TIMESTAMP,
    ER_DISCHARGE_TIMESTAMP,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    LENGTH_OF_STAY_IN_ER,
    SCU_ADMIT_TIMESTAMP,
    SCU_DISCHARGE_TIMESTAMP,
    SEX,
)
from cyclops.processors.constants import EMPTY_STRING, MONTH, YEAR
from cyclops.query.interface import QueryInterface
from cyclops.query.util import (
    DBTable,
    equals,
    get_attribute,
    has_substring,
    in_,
    not_equals,
    rename_attributes,
    rga,
    to_datetime_format,
    to_list,
)

IP_ADMIN = "ip_admin"
ER_ADMIN = "er_admin"
DIAGNOSIS = "diagnosis"
LAB = "lab"
VITALS = "vitals"
PHARMACY = "pharmacy"
INTERVENTION = "intervention"
LOOKUP_IP_ADMIN = "lookup_ip_admin"
LOOKUP_ER_ADMIN = "lookup_er_admin"
LOOKUP_DIAGNOSIS = "lookup_diagnosis"
IP_SCU = "ip_scu"
LOOKUP_ROOM_TRANSFER = "lookup_room_transfer"
ROOM_TRANSFER = "room_transfer"

_db = Database(config.read_config(GEMINI))
TABLE_MAP = {
    IP_ADMIN: lambda db: db.public.ip_administrative,
    ER_ADMIN: lambda db: db.public.er_administrative,
    DIAGNOSIS: lambda db: db.public.diagnosis,
    LAB: lambda db: db.public.lab,
    VITALS: lambda db: db.public.vitals,
    PHARMACY: lambda db: db.public.pharmacy,
    INTERVENTION: lambda db: db.public.intervention,
    LOOKUP_IP_ADMIN: lambda db: db.public.lookup_ip_administrative,
    LOOKUP_ER_ADMIN: lambda db: db.public.lookup_er_administrative,
    LOOKUP_DIAGNOSIS: lambda db: db.public.lookup_diagnosis,
    IP_SCU: lambda db: db.public.ip_scu,
    ROOM_TRANSFER: lambda db: db.public.room_transfer,
    LOOKUP_ROOM_TRANSFER: lambda db: db.public.lookup_room_transfer,
}
GEMINI_COLUMN_MAP = {
    "genc_id": ENCOUNTER_ID,
    "admit_date_time": ADMIT_TIMESTAMP,
    "discharge_date_time": DISCHARGE_TIMESTAMP,
    "gender": SEX,
    "result_value": EVENT_VALUE,
    "result_unit": EVENT_VALUE_UNIT,
    "lab_test_name_mapped": EVENT_NAME,
    "sample_collection_date_time": EVENT_TIMESTAMP,
    "measurement_mapped": EVENT_NAME,
    "measurement_value": EVENT_VALUE,
    "measure_date_time": EVENT_TIMESTAMP,
    "left_er_date_time": ER_DISCHARGE_TIMESTAMP,
    "duration_er_stay_derived": LENGTH_OF_STAY_IN_ER,
    "triage_date_time": ER_ADMIT_TIMESTAMP,
    "er_discharge_timestamp": ER_DISCHARGE_TIMESTAMP,
    "scu_admit_date_time": SCU_ADMIT_TIMESTAMP,
    "scu_discharge_date_time": SCU_DISCHARGE_TIMESTAMP,
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
    if table_name not in [LOOKUP_IP_ADMIN, LOOKUP_ER_ADMIN, LOOKUP_DIAGNOSIS]:
        raise ValueError("Not a recognised lookup/dimension table!")

    subquery = select(TABLE_MAP[table_name](_db).data).subquery()

    return QueryInterface(_db, subquery)


def _er_admin() -> QueryInterface:
    """Query emergency room administrative data.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table = TABLE_MAP[ER_ADMIN](_db)
    table = rename_attributes(table, GEMINI_COLUMN_MAP)
    return QueryInterface(_db, table)


def _map_table_attributes(table_name: str) -> Subquery:
    """Map queried data attributes into common column names.

    For unifying processing functions across datasets, data
    attributes are currently mapped to the same name, to allow for processing
    to recognise them.

    Parameters
    ----------
    table_name: str
        Name of GEMINI table.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Query with mapped attributes.

    """
    return rename_attributes(TABLE_MAP[table_name](_db).data, GEMINI_COLUMN_MAP)


def rename_timestamps(
    table: Union[Select, Subquery, Table, DBTable],
    admit_ts_col: str,
    discharge_ts_col: str,
    care_unit_name: str,
) -> Select:
    """Rename timestamp columns from different tables.

    Parameters
    ----------
    table: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or
    cyclops.query.util.DBTable

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        Query with renamed timestamp columns.

    """
    return select(
        rga(table.c, ENCOUNTER_ID),
        rga(table.c, admit_ts_col).label("admit"),
        rga(table.c, discharge_ts_col).label("discharge"),
        literal(care_unit_name).label("care_unit_name"),
    )


def patients(  # pylint: disable=too-many-arguments
    years: Optional[Union[int, List[int]]] = None,
    months: Optional[Union[int, List[int]]] = None,
    hospitals: Optional[Union[str, List[str]]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    delirium_cohort: Optional[bool] = False,
    include_er_data: Optional[bool] = False,
) -> QueryInterface:
    """Query patient encounters.

    Parameters
    ----------
    years: int or list of int, optional
        Years for which patient encounters are to be filtered.
    months: int or list of int, optional
        Months for which patient encounters are to be filtered.
    hospitals: str or list of str, optional
        Hospital sites from which patient encounters are to be filtered.
    from_date: str, optional
        Gather patients admitted >= from_date in YYYY-MM-DD format.
    to_date: str, optional
        Gather patients admitted <= to_date in YYYY-MM-DD format.
    delirium_cohort: bool, optional
        Gather patient encounters for which delirium label is available.
    include_er_data: bool, optional
        Gather Emergency Room data recorded for the particular encounter.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[IP_ADMIN](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP)

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
    if hospitals:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.hospital_id, to_list(hospitals), to_str=True))
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
    if delirium_cohort:
        subquery = (
            select(subquery).where(equals(subquery.c.gemini_cohort, True)).subquery()
        )
    if include_er_data:
        er_subquery = _er_admin().query
        subquery = (
            select(subquery, er_subquery)
            .where(subquery.c.encounter_id == er_subquery.c.encounter_id)
            .subquery()
        )

    return QueryInterface(_db, subquery)


def _join_with_patients(
    patients_table: Union[Select, Subquery, Table, DBTable],
    table_: Union[Select, Subquery, Table, DBTable],
) -> QueryInterface:
    """Join a subquery with GEMINI patient static information.

    Assumes that both query tables have "encounter_id".

    Parameters
    ----------
    patients_table: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or
    cyclops.query.util.DBTable
        Patient query table.
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.util.DBTable
        A query table such as labs or vitals.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    if not hasattr(table_.c, ENCOUNTER_ID) or not hasattr(
        patients_table.c, ENCOUNTER_ID
    ):
        raise ValueError("Input query table and patients table must have encounter_id!")

    # Join on patients (encounter_id).
    query = select(table_, patients_table).where(
        table_.c.encounter_id == patients_table.c.encounter_id
    )
    return QueryInterface(_db, query)


def diagnoses(
    diagnosis_codes: Union[List[str], str] = None,
    diagnosis_types: List[str] = None,
    patients: Optional[QueryInterface] = None,  # pylint: disable=redefined-outer-name
) -> QueryInterface:
    """Query diagnosis data.

    Parameters
    ----------
    diagnosis_codes: list of str or str, optional
        Names of diagnosis codes to include, or a diagnosis code search string,
        all diagnosis data are included if not provided.
    diagnosis_types: list of str, optional
        Include only those diagnoses that are of certain type.
    patients: QueryInterface, optional
        Patient encounters query wrapped, used to join with diagnoses.

    Notes
    -----
        Refer to the diagnosis lookup table for descriptions for diagnosis types.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table_ = TABLE_MAP[DIAGNOSIS](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP)
    if diagnosis_codes:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.diagnosis_code, diagnosis_codes, to_str=True))
            .subquery()
        )
    if diagnosis_types:
        subquery = (
            select(subquery)
            .where(in_(subquery.c.diagnosis_type, diagnosis_types, to_str=True))
            .subquery()
        )
    if patients:
        return _join_with_patients(patients.query, subquery)

    return QueryInterface(_db, subquery)


def get_careunits(encounters: list = None) -> Subquery:
    """Get care unit table within a given set of encounters.

    Parameters
    ----------
    encounters : list
        The encounter IDs to consider. If None, consider all encounters.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Constructed query, wrapped in an interface object.

    """
    filter_encounters = (
        lambda query, encounters: query
        if encounters is None
        else select(query)
        .where(in_(get_attribute(query, ENCOUNTER_ID), encounters))
        .subquery()
    )

    # In-patient table.
    ip_table = filter_encounters(_map_table_attributes(IP_ADMIN), encounters)
    ip_table = rename_timestamps(ip_table, ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP, "IP")

    # Special care unit table.
    scu_table = filter_encounters(_map_table_attributes(IP_SCU), encounters)
    scu_table = rename_timestamps(
        scu_table, SCU_ADMIT_TIMESTAMP, SCU_DISCHARGE_TIMESTAMP, "SCU"
    )

    # Emergency room/department table.
    er_table = filter_encounters(_er_admin().query, encounters)
    er_table = rename_timestamps(
        er_table, ER_ADMIT_TIMESTAMP, ER_DISCHARGE_TIMESTAMP, "ER"
    )

    # Room transfer table.
    rt_table = filter_encounters(_map_table_attributes(ROOM_TRANSFER), encounters)
    lookup_rt_table = _map_table_attributes(LOOKUP_ROOM_TRANSFER)

    # Join room transfer with lookup to get description.
    rt_table = select(
        rga(rt_table.c, ENCOUNTER_ID),
        rt_table.c.checkin_date_time.label("admit"),
        rt_table.c.checkout_date_time.label("discharge"),
        lookup_rt_table.c.care_unit_name,
    ).where(rt_table.c.medical_service == lookup_rt_table.c.value)

    return QueryInterface(_db, union_all(rt_table, ip_table, scu_table, er_table))


def events(
    category: str,
    names: Optional[Union[str, List[str]]] = None,
    substring: Optional[str] = None,
    patients: Optional[QueryInterface] = None,  # pylint: disable=redefined-outer-name
) -> QueryInterface:
    """Query events.

    Parameters
    ----------
    category : str
        Category of events i.e. lab, vitals, intervention, etc.
    names : str or list of str, optional
        The event name(s) to filter.
    substring: str, optional
        Substring to search event names to filter.
    patients: QueryInterface, optional
        Patient encounters query wrapped, used to join with events.
    care_unit: bool
        Whether to include the care unit for each event.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    if category not in [LAB, VITALS, INTERVENTION]:
        raise ValueError("Invalid category of events specified!")
    table_ = TABLE_MAP[category](_db)
    subquery = select(table_.data)
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP)

    if category != INTERVENTION:
        subquery = (
            select(subquery)
            .where(not_equals(subquery.c.event_name, EMPTY_STRING, to_str=True))
            .subquery()
        )
        if names:
            subquery = (
                select(subquery)
                .where(in_(subquery.c.event_name, to_list(names), to_str=True))
                .subquery()
            )
        if substring:
            subquery = (
                select(subquery)
                .where(has_substring(subquery.c.event_name, substring, to_str=True))
                .subquery()
            )

    if patients:
        return _join_with_patients(patients.query, subquery)

    return QueryInterface(_db, subquery)
