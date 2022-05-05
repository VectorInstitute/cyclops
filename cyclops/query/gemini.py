"""GEMINI query API."""

from typing import List, Optional, Union

from sqlalchemy import extract, select
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops import config
from cyclops.constants import GEMINI
from cyclops.orm import Database
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    CARE_UNIT,
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
    get_attributes,
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
    ROOM_TRANSFER: lambda db: db.public.ip_scu,
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
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()

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
        er_table = TABLE_MAP[ER_ADMIN](_db)
        er_subquery = select(er_table.data)
        er_subquery = rename_attributes(er_subquery, GEMINI_COLUMN_MAP).subquery()
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
    sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
        Patient query table.
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
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
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()
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


def _include_careunit(events: Subquery) -> Subquery:
    """Creates care unit column in events table.
    
    Parameters
    ----------
    events : sqlalchemy.sql.selectable.Subquery
        The events table.
    Returns
    sqlalchemy.sql.selectable.Subquery
        The events table with added attributed CARE_UNIT
    -------
    """
    
    ip_table = TABLE_MAP[IP_ADMIN](_db).data
    ip_table = rename_attributes(ip_table, GEMINI_COLUMN_MAP)
    ip_table = get_attributes([ENCOUNTER_ID, ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP]).subquery()
    
    er_table = TABLE_MAP[ER_ADMIN](_db).data
    er_table = rename_attributes(er_table, GEMINI_COLUMN_MAP)
    er_table = get_attributes([ER_DISCHARGE_TIMESTAMP, LENGTH_OF_STAY_IN_ER])
    er_table = select(
        er_table,
        (rga(er_table.c, ER_DISCHARGE_TIMESTAMP) - rga(er_table.c, LENGTH_OF_STAY_IN_ER)).label(ER_ADMIT_TIMESTAMP)
    ).subquery()
    
    scu_table = TABLE_MAP[IP_SCU](_db).data
    scu_table = rename_attributes(scu_table, GEMINI_COLUMN_MAP)  
    scu_table = get_attributes([ENCOUNTER_ID, SCU_ADMIT_TIMESTAMP, SCU_DISCHARGE_TIMESTAMP])
    
    # ER = Emergency?
    # ICU = SCU?
    
    rt_table = TABLE_MAP[ROOM_TRANSFER](_db).data
    rt_table = rename_attributes(rt_table, GEMINI_COLUMN_MAP)
    
    lookup_rt_table = TABLE_MAP[LOOKUP_ROOM_TRANSFER](_db).data
    lookup_rt_table = rename_attributes(lookup_rt_table, GEMINI_COLUMN_MAP)
    
    # Join room transfer with lookup to get description
    rt_table = select(rga(rt_table, ENCOUNTER_ID), lookup_rt_table.description).where(
        rt_table.medical_service == lookup_rt_table.value
    )
    
    #care_unit = ... .label(CARE_UNIT)
    #events ... join... care_unit ... on ENCOUNTER_ID
    #return select(events, .label(CARE_UNIT)).subquery()
    return ip_table, er_table, scu_table, rt_table


def events(
    category: str,
    names: Optional[Union[str, List[str]]] = None,
    substring: Optional[str] = None,
    patients: Optional[QueryInterface] = None,
    care_unit = True # pylint: disable=redefined-outer-name
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
    subquery = rename_attributes(subquery, GEMINI_COLUMN_MAP).subquery()

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

    if care_unit:
        subquery = careunit(subquery)
            
    if patients:
        return _join_with_patients(patients.query, subquery)

    return QueryInterface(_db, subquery)
