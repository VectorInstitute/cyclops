"""OMOP Query API."""

import logging
from typing import Dict, List, Optional, Union

from sqlalchemy.sql.selectable import Subquery

from cyclops.query import process as qp
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface
from cyclops.query.util import TableTypes, table_params_to_type
from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# Table names.
VISIT_OCCURRENCE = "visit_occurrence"
VISIT_DETAIL = "visit_detail"
PERSON = "person"
MEASUREMENT = "measurement"
CONCEPT = "concept"
OBSERVATION = "observation"
CARE_SITE = "care_site"

# OMOP column names.
VISIT_OCCURRENCE_ID = "visit_occurrence_id"
PERSON_ID = "person_id"
VISIT_START_DATETIME = "visit_start_datetime"
VISIT_END_DATETIME = "visit_end_datetime"
VISIT_DETAIL_START_DATETIME = "visit_detail_start_datetime"
VISIT_DETAIL_END_DATETIME = "visit_detail_end_datetime"
VISIT_CONCEPT_ID = "visit_concept_id"
VISIT_TYPE_CONCEPT_ID = "visit_type_concept_id"
VISIT_DETAIL_CONCEPT_ID = "visit_detail_concept_id"
VISIT_DETAIL_TYPE_CONCEPT_ID = "visit_detail_type_concept_id"
CARE_SITE_ID = "care_site_id"
CONCEPT_NAME = "concept_name"
CONCEPT_ID = "concept_id"
CARE_SITE_SOURCE_VALUE = "care_site_source_value"
OBSERVATION_CONCEPT_ID = "observation_concept_id"
OBSERVATION_TYPE_CONCEPT_ID = "observation_type_concept_id"
OBSERVATION_DATETIME = "observation_datetime"
MEASUREMENT_CONCEPT_ID = "measurement_concept_id"
MEASUREMENT_TYPE_CONCEPT_ID = "measurement_type_concept_id"
MEASUREMENT_DATETIME = "measurement_datetime"
UNIT_CONCEPT_ID = "unit_concept_id"
VALUE_AS_CONCEPT_ID = "value_as_concept_id"

# Created columns.
VISIT_DETAIL_CONCEPT_NAME = "visit_detail_concept_name"
CARE_SITE_NAME = "care_site_name"
GENDER_CONCEPT_NAME = "gender_concept_name"
RACE_CONCEPT_NAME = "race_concept_name"
ETHNICITY_CONCEPT_NAME = "ethnicity_concept_name"

# Column map.
COLUMN_MAP: dict = {}

# Other constants
ID = "id"
NAME = "name"


def _get_table_map(schema_name: str) -> Dict:
    """Get table map.

    Parameters
    ----------
    schema_name: str
        Name of schema.

    Returns
    -------
    Dict
        A mapping of table names to the ORM table objects.

    """
    return {
        VISIT_OCCURRENCE: lambda db: getattr(db, schema_name).visit_occurrence,
        VISIT_DETAIL: lambda db: getattr(db, schema_name).visit_detail,
        PERSON: lambda db: getattr(db, schema_name).person,
        MEASUREMENT: lambda db: getattr(db, schema_name).measurement,
        OBSERVATION: lambda db: getattr(db, schema_name).observation,
        CONCEPT: lambda db: getattr(db, schema_name).concept,
        CARE_SITE: lambda db: getattr(db, schema_name).care_site,
    }


class OMOPQuerier(DatasetQuerier):
    """OMOP querier."""

    def __init__(
        self,
        schema_name: str,
        config_overrides: Optional[List] = None,
    ):
        """Initialize.

        Parameters
        ----------
        config_overrides: list, optional
            List of override configuration parameters.

        """
        if not config_overrides:
            config_overrides = []
        super().__init__(_get_table_map(schema_name), COLUMN_MAP, config_overrides)

    def _map_concept_ids_to_name(
        self, source_table: Subquery, source_cols: Union[str, List[str]]
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
        concept_table = self.get_table(CONCEPT)
        for col in to_list(source_cols):
            if ID not in col:
                raise ValueError("Specified column not a concept ID column!")
            source_table = qp.Join(
                concept_table,
                on=(col, CONCEPT_ID),
                join_table_cols=[CONCEPT_NAME],
                isouter=True,
            )(source_table)
            source_table = qp.Rename({CONCEPT_NAME: col.replace(ID, NAME)})(
                source_table
            )

        return source_table

    def _map_care_site_id(self, source_table: Subquery) -> Subquery:
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
        care_site_table = self.get_table(CARE_SITE)
        source_table = qp.Join(
            care_site_table,
            on=CARE_SITE_ID,
            join_table_cols=[CARE_SITE_NAME, CARE_SITE_SOURCE_VALUE],
            isouter=True,
        )(source_table)

        return source_table

    @table_params_to_type(Subquery)
    def visit_occurrence(
        self,
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
        table = self.get_table(VISIT_OCCURRENCE)

        if drop_null_person_ids:
            table = qp.DropNulls(PERSON_ID)(table)

        # Possibly cast string representations to timestamps.
        table = qp.Cast([VISIT_START_DATETIME], "timestamp")(table)

        # Map concept IDs to concept table cols.
        table = self._map_concept_ids_to_name(
            table, ["visit_concept_id", "visit_type_concept_id"]
        )

        # Map care_site ID to care_site information from care_site table.
        table = self._map_care_site_id(table)

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

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    def visit_detail(
        self,
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
        table = self.get_table(VISIT_DETAIL)

        # Possibly cast string representations to timestamps
        table = qp.Cast([VISIT_DETAIL_START_DATETIME], "timestamp")(table)

        if visit_occurrence_table is not None:
            table = qp.Join(
                visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID]
            )(table)

        table = self._map_concept_ids_to_name(
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
            (
                qp.ConditionSubstring,
                [VISIT_DETAIL_CONCEPT_NAME, qp.QAP("care_unit")],
                {},
            ),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]

        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    def person(
        self,
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
        table = self.get_table(PERSON)

        if visit_occurrence_table is not None:
            table = qp.Join(visit_occurrence_table, on=PERSON_ID)(table)

        table = self._map_concept_ids_to_name(
            table, ["gender_concept_id", "race_concept_id", "ethnicity_concept_id"]
        )

        operations: List[tuple] = [
            (qp.ConditionIn, [GENDER_CONCEPT_NAME, qp.QAP("gender")], {}),
            (qp.ConditionIn, [RACE_CONCEPT_NAME, qp.QAP("race")], {}),
            (qp.ConditionIn, [ETHNICITY_CONCEPT_NAME, qp.QAP("ethnicity")], {}),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]

        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    def observation(
        self,
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
        table = self.get_table(OBSERVATION)

        if visit_occurrence_table is not None:
            table = qp.Join(
                visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID]
            )(table)

        # Possibly cast string representations to timestamps
        table = qp.Cast([OBSERVATION_DATETIME], "timestamp")(table)

        table = self._map_concept_ids_to_name(
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

        return QueryInterface(self._db, table)

    @table_params_to_type(Subquery)
    def measurement(
        self,
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
        table = self.get_table(MEASUREMENT)

        if visit_occurrence_table is not None:
            table = qp.Join(
                visit_occurrence_table, on=[PERSON_ID, VISIT_OCCURRENCE_ID]
            )(table)

        # Possibly cast string representations to timestamps
        table = qp.Cast([MEASUREMENT_DATETIME], "timestamp")(table)

        # Cast value_as_concept_id to int.
        table = qp.Cast([VALUE_AS_CONCEPT_ID], "int")(table)

        table = self._map_concept_ids_to_name(
            table,
            [MEASUREMENT_CONCEPT_ID, MEASUREMENT_TYPE_CONCEPT_ID, UNIT_CONCEPT_ID],
        )

        operations: List[tuple] = [
            (qp.ConditionBeforeDate, [MEASUREMENT_DATETIME, qp.QAP("before_date")], {}),
            (qp.ConditionAfterDate, [MEASUREMENT_DATETIME, qp.QAP("after_date")], {}),
            (qp.ConditionInYears, [MEASUREMENT_DATETIME, qp.QAP("years")], {}),
            (qp.ConditionInMonths, [MEASUREMENT_DATETIME, qp.QAP("months")], {}),
            (qp.Limit, [qp.QAP("limit")], {}),
        ]

        table = qp.process_operations(table, operations, process_kwargs)

        return QueryInterface(self._db, table)
