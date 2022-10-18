"""GEMINI OMOP query API functions."""

import logging
from typing import Callable, List, Optional, Union

from sqlalchemy.sql.selectable import Subquery

from codebase_ops import get_log_file_path
from cyclops import config
from cyclops.constants import GEMINI_OMOP
from cyclops.orm import Database
from cyclops.query import process as qp
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.omop import (
    CARE_SITE,
    CARE_SITE_ID,
    CARE_SITE_NAME,
    CARE_SITE_SOURCE_VALUE,
    CONCEPT,
    CONCEPT_ID,
    CONCEPT_NAME,
    ETHNICITY_CONCEPT_NAME,
    GENDER_CONCEPT_NAME,
    MEASUREMENT,
    MEASUREMENT_CONCEPT_ID,
    MEASUREMENT_DATETIME,
    MEASUREMENT_TYPE_CONCEPT_ID,
    OBSERVATION,
    OBSERVATION_CONCEPT_ID,
    OBSERVATION_DATETIME,
    OBSERVATION_TYPE_CONCEPT_ID,
    OMOP_COLUMN_MAP,
    PERSON,
    PERSON_ID,
    RACE_CONCEPT_NAME,
    TABLE_MAP,
    UNIT_CONCEPT_ID,
    VALUE_AS_CONCEPT_ID,
    VISIT_DETAIL,
    VISIT_DETAIL_CONCEPT_NAME,
    VISIT_DETAIL_START_DATETIME,
    VISIT_OCCURRENCE,
    VISIT_OCCURRENCE_ID,
    VISIT_START_DATETIME,
)
from cyclops.query.util import TableTypes, _to_subquery, table_params_to_type
from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


_db = Database(config.read_config(GEMINI_OMOP))


# Constants
ID = "id"
NAME = "name"


@table_params_to_type(Subquery)
def get_interface(
    table: TableTypes,
    process_fn: Optional[Callable] = None,
) -> Union[QueryInterface, QueryInterfaceProcessed]:
    """Get a query interface for an OMOP table.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        Table to wrap in the interface.
    process_fn: Callable
        Process function to apply on the Pandas DataFrame returned from the query.

    Returns
    -------
    cyclops.query.interface.QueryInterface or
    cyclops.query.interface.QueryInterfaceProcessed
        A query interface using the GEMINI database object.

    """
    if process_fn is None:
        return QueryInterface(_db, table)

    return QueryInterfaceProcessed(_db, table, process_fn)


def get_table(table_name: str, rename: bool = True) -> Subquery:
    """Get a table and possibly map columns to have standard names.

    Standardizing column names allows for for columns to be
    recognized in downstream processing.

    Parameters
    ----------
    table_name: str
        Name of table.
    rename: bool, optional
        Whether to map the column names

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Table with mapped columns.

    """
    if table_name not in TABLE_MAP:
        raise ValueError(f"{table_name} not a recognised table.")

    table = TABLE_MAP[table_name](_db).data

    if rename:
        table = qp.Rename(OMOP_COLUMN_MAP, check_exists=False)(table)

    return _to_subquery(table)


def _map_concept_ids_to_name(
    source_table: Subquery, source_cols: Union[str, List[str]]
) -> Subquery:
    """Map concept IDs in a source table to concept names from concept table.

    Parameters
    ----------
    source_table: Subquery
        Source table with concept IDs.
    source_cols: list of str
        List of columns names to consider as concept IDs for mapping.

    Returns
    -------
    Subquery
        Query with mapped columns from concept table.

    """
    concept_table = get_table(CONCEPT)
    for col in to_list(source_cols):
        if ID not in col:
            raise ValueError("Specified column not a concept ID column!")
        source_table = qp.Join(
            concept_table, on=(col, CONCEPT_ID), join_table_cols=[CONCEPT_NAME]
        )(source_table)
        source_table = qp.Rename({CONCEPT_NAME: col.replace(ID, NAME)})(source_table)

    return source_table


def _map_care_site_id(source_table: Subquery) -> Subquery:
    """Map care_site_id in a source table to care_site table.

    Parameters
    ----------
    source_table: Subquery
        Source table with care_site_id.

    Returns
    -------
    Subquery
        Query with mapped columns from care_site table.

    """
    care_site_table = get_table(CARE_SITE)
    source_table = qp.Join(
        care_site_table,
        on=CARE_SITE_ID,
        join_table_cols=[CARE_SITE_NAME, CARE_SITE_SOURCE_VALUE],
    )(source_table)

    return source_table


@table_params_to_type(Subquery)
def visit_occurrence(
    drop_null_person_ids=True,
    **process_kwargs,
) -> QueryInterface:
    """Query OMOP visit_occurrence table.

    Parameters
    ----------
    drop_null_person_ids: bool, optional
        Flag to say if entries should be dropped if 'person_id' is missing.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    before_date: datetime.datetime or str
        Get patient visits starting before some date.
        If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patient visits starting after some date.
        If a string, provide in YYYY-MM-DD format.
    hospitals: str or list of str, optional
        Get patient visits by hospital sites.
    years: int or list of int, optional
        Get patient visits by year.
    months: int or list of int, optional
        Get patient visits by month.
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(VISIT_OCCURRENCE)

    if drop_null_person_ids:
        table = qp.DropNulls(PERSON_ID)(table)

    # Possibly cast string representations to timestamps.
    table = qp.Cast([VISIT_START_DATETIME], "timestamp")(table)

    # Map concept IDs to concept table cols.
    table = _map_concept_ids_to_name(
        table, ["visit_concept_id", "visit_type_concept_id"]
    )

    # Map care_site ID to care_site information from care_site table.
    table = _map_care_site_id(table)

    operations: List[tuple] = [
        (qp.ConditionBeforeDate, [VISIT_START_DATETIME, qp.QAP("before_date")], {}),
        (qp.ConditionAfterDate, [VISIT_START_DATETIME, qp.QAP("after_date")], {}),
        (qp.ConditionInYears, [VISIT_START_DATETIME, qp.QAP("years")], {}),
        (qp.ConditionInMonths, [VISIT_START_DATETIME, qp.QAP("months")], {}),
        (
            qp.ConditionIn,
            [CARE_SITE_SOURCE_VALUE, qp.QAP("hospitals")],
            {"to_str": True},
        ),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
def visit_detail(
    visit_occurrence_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query OMOP visit_detail table.

    Parameters
    ----------
    visit_occurrence_table: Subquery, optional
        Visit occurrence table to join on.

    Other Parameters
    ----------------
    before_date: datetime.datetime or str
        Get patient visits starting before some date.
        If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patient visits starting after some date.
        If a string, provide in YYYY-MM-DD format.
    years: int or list of int, optional
        Get patient visits by year.
    months: int or list of int, optional
        Get patient visits by month.
    care_unit: str or list of str
        Filter on care_unit, accepts substring e.g. "Emergency Room".
    limit: int, optional
        Limit the number of rows returned.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table = get_table(VISIT_DETAIL)

    # Possibly cast string representations to timestamps
    table = qp.Cast([VISIT_DETAIL_START_DATETIME], "timestamp")(table)

    if visit_occurrence_table is not None:
        table = qp.Join(visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID])(
            table
        )

    table = _map_concept_ids_to_name(
        table, ["visit_detail_concept_id", "visit_detail_type_concept_id"]
    )

    operations: List[tuple] = [
        (
            qp.ConditionBeforeDate,
            [VISIT_DETAIL_START_DATETIME, qp.QAP("before_date")],
            {},
        ),
        (
            qp.ConditionAfterDate,
            [VISIT_DETAIL_START_DATETIME, qp.QAP("after_date")],
            {},
        ),
        (qp.ConditionInYears, [VISIT_DETAIL_START_DATETIME, qp.QAP("years")], {}),
        (qp.ConditionInMonths, [VISIT_DETAIL_START_DATETIME, qp.QAP("months")], {}),
        (qp.ConditionSubstring, [VISIT_DETAIL_CONCEPT_NAME, qp.QAP("care_unit")], {}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
def person(
    visit_occurrence_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query OMOP person table.

    Parameters
    ----------
    visit_occurrence_table: Subquery, optional
        Visit occurrence table to join on.

    Other Parameters
    ----------------
    gender: str or list of str
        Filter on gender.
    race: str or list of str
        Filter on race.
    ethnicity: str or list of str
        Filter on ethnicity.
    limit: int, optional
        Limit the number of rows returned.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table = get_table(PERSON)

    if visit_occurrence_table is not None:
        table = qp.Join(visit_occurrence_table, on=PERSON_ID)(table)

    table = _map_concept_ids_to_name(
        table, ["gender_concept_id", "race_concept_id", "ethnicity_concept_id"]
    )

    operations: List[tuple] = [
        (qp.ConditionIn, [GENDER_CONCEPT_NAME, qp.QAP("gender")], {}),
        (qp.ConditionIn, [RACE_CONCEPT_NAME, qp.QAP("race")], {}),
        (qp.ConditionIn, [ETHNICITY_CONCEPT_NAME, qp.QAP("ethnicity")], {}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
def observation(
    visit_occurrence_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query OMOP observation table.

    Parameters
    ----------
    visit_occurrence_table: Subquery, optional
        Visit occurrence table to join on.

    Other Parameters
    ----------------
    before_date: datetime.datetime or str
        Get patient observations starting before some date.
        If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patient observations starting after some date.
        If a string, provide in YYYY-MM-DD format.
    years: int or list of int, optional
        Get patient observations by year.
    months: int or list of int, optional
        Get patient observations by month.
    limit: int, optional
        Limit the number of rows returned.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table = get_table(OBSERVATION)

    if visit_occurrence_table is not None:
        table = qp.Join(visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID])(
            table
        )

    # Possibly cast string representations to timestamps
    table = qp.Cast([OBSERVATION_DATETIME], "timestamp")(table)

    table = _map_concept_ids_to_name(
        table, [OBSERVATION_CONCEPT_ID, OBSERVATION_TYPE_CONCEPT_ID]
    )

    operations: List[tuple] = [
        (qp.ConditionBeforeDate, [OBSERVATION_DATETIME, qp.QAP("before_date")], {}),
        (qp.ConditionAfterDate, [OBSERVATION_DATETIME, qp.QAP("after_date")], {}),
        (qp.ConditionInYears, [OBSERVATION_DATETIME, qp.QAP("years")], {}),
        (qp.ConditionInMonths, [OBSERVATION_DATETIME, qp.QAP("months")], {}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
def measurement(
    visit_occurrence_table: Optional[TableTypes] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query OMOP measurement table.

    Parameters
    ----------
    visit_occurrence_table: Subquery, optional
        Visit occurrence table to join on.

    Other Parameters
    ----------------
    before_date: datetime.datetime or str
        Get patient measurements starting before some date.
        If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patient measurements starting after some date.
        If a string, provide in YYYY-MM-DD format.
    years: int or list of int, optional
        Get patient measurements by year.
    months: int or list of int, optional
        Get patient measurements by month.
    limit: int, optional
        Limit the number of rows returned.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table = get_table(MEASUREMENT)

    if visit_occurrence_table is not None:
        table = qp.Join(visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID])(
            table
        )

    # Possibly cast string representations to timestamps
    table = qp.Cast([MEASUREMENT_DATETIME], "timestamp")(table)

    # Cast value_as_concept_id to int.
    table = qp.Cast([VALUE_AS_CONCEPT_ID], "int")(table)

    table = _map_concept_ids_to_name(
        table, [MEASUREMENT_CONCEPT_ID, MEASUREMENT_TYPE_CONCEPT_ID, UNIT_CONCEPT_ID]
    )

    operations: List[tuple] = [
        (qp.ConditionBeforeDate, [MEASUREMENT_DATETIME, qp.QAP("before_date")], {}),
        (qp.ConditionAfterDate, [MEASUREMENT_DATETIME, qp.QAP("after_date")], {}),
        (qp.ConditionInYears, [MEASUREMENT_DATETIME, qp.QAP("years")], {}),
        (qp.ConditionInMonths, [MEASUREMENT_DATETIME, qp.QAP("months")], {}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)

    return QueryInterface(_db, table)
