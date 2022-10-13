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
    CARE_SITE_ID,
    OMOP_COLUMN_MAP,
    PERSON_ID,
    VISIT_OCCURRENCE_ID,
    TABLE_MAP,
    VISIT_END_DATETIME,
    VISIT_OCCURRENCE,
    VISIT_DETAIL,
    VISIT_START_DATETIME,
    CONCEPT,
    CONCEPT_NAME,
    CONCEPT_ID,
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
    """Get a query interface for a GEMINI_OMOP table.

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
        Name of GEMINI_OMOP table.
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


def _map_concept_ids_to_name(source_table: Subquery, source_cols: List[str]) -> Subquery:
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
        source_table = qp.Join(concept_table, on=(col, CONCEPT_ID), join_table_cols=[CONCEPT_NAME])(source_table)
        source_table = qp.Rename({CONCEPT_NAME: col.replace(ID, NAME)})(source_table)
        
    return source_table

@table_params_to_type(Subquery)
def visit_occurrence(
    drop_null_person_ids=True,
    map_concept_cols: Union[None, str, List[str]] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query GEMINI_OMOP visit_occurrence table.

    Parameters
    ----------
    drop_null_person_ids: bool, optional
        Flag to say if entries should be dropped if 'person_id' is missing.
    map_concept_cols: str or list of str, optional
        Names of columns which need to be joined with concept table
        to map concepts (concept names are created as new column).
    

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    Other Parameters
    ----------------
    before_date: datetime.datetime or str
        Get patients encounters before some date.
        If a string, provide in YYYY-MM-DD format.
    after_date: datetime.datetime or str
        Get patients encounters after some date.
        If a string, provide in YYYY-MM-DD format.
    hospitals: str or list of str, optional
        Get patient encounters by hospital sites.
    years: int or list of int, optional
        Get patient encounters by year.
    months: int or list of int, optional
        Get patient encounters by month.
    limit: int, optional
        Limit the number of rows returned.

    """
    table = get_table(VISIT_OCCURRENCE)

    if drop_null_person_ids:
        table = qp.DropNulls(PERSON_ID)(table)

    # Possibly cast string representations to timestamps
    table = qp.Cast([VISIT_START_DATETIME, VISIT_END_DATETIME], "timestamp")(table)

    operations: List[tuple] = [
        (qp.ConditionBeforeDate, [VISIT_START_DATETIME, qp.QAP("before_date")], {}),
        (qp.ConditionAfterDate, [VISIT_START_DATETIME, qp.QAP("after_date")], {}),
        (qp.ConditionInYears, [VISIT_START_DATETIME, qp.QAP("years")], {}),
        (qp.ConditionInMonths, [VISIT_START_DATETIME, qp.QAP("months")], {}),
        (qp.ConditionIn, [CARE_SITE_ID, qp.QAP("hospitals")], {"to_str": True}),
        (qp.Limit, [qp.QAP("limit")], {}),
    ]

    table = qp.process_operations(table, operations, process_kwargs)
    
    if map_concept_cols is not None:       
        table = _map_concept_ids_to_name(table, map_concept_cols)
    
    return QueryInterface(_db, table)


@table_params_to_type(Subquery)
def visit_detail(
    visit_occurrence_table: Optional[TableTypes] = None,
    map_concept_cols: Union[None, str, List[str]] = None,
    **process_kwargs,
) -> QueryInterface:
    """Query GEMINI_OMOP visit_detail table.

    Parameters
    ----------
    visit_occurrence_table: Subquery, optional
        Visit occurrence table to join on.
    map_concept_cols: str or list of str, optional
        Names of columns which need to be joined with concept table
        to map concepts (concept names are created as new column).

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    table = get_table(VISIT_DETAIL)

    if visit_occurrence_table is not None:
        table = qp.Join(visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID])(table)
        
    if map_concept_cols is not None:       
        table = _map_concept_ids_to_name(table, map_concept_cols)

    return QueryInterface(_db, table)
