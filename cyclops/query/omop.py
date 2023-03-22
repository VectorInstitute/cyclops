"""OMOP query API."""

import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.sql.selectable import Subquery

import cyclops.query.ops as qo
from cyclops.query.base import DatasetQuerier
from cyclops.query.interface import QueryInterface
from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


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

# Other constants
ID = "id"
NAME = "name"


class OMOPQuerier(DatasetQuerier):
    """OMOP querier."""

    def __init__(
        self,
        schema_name: str,
        **config_overrides: Dict[str, Any],
    ) -> None:
        """Initialize.

        Parameters
        ----------
        schema_name: str
            Name of database schema.
        **config_overrides
            Override configuration parameters, specified as kwargs.

        """
        self.schema_name = schema_name
        overrides = {}
        if config_overrides:
            overrides = config_overrides
        super().__init__(**overrides)

    def map_concept_ids_to_name(
        self,
        src_table: Subquery,
        src_cols: Union[str, List[str]],
        dst_cols: Optional[Union[str, List[str]]] = None,
    ) -> Subquery:
        """Map concept IDs in a source table to concept names from concept table.

        For each concept ID column with a name like `somecol_concept_ID`, the mapped
        concept name column will be named `somecol_concept_name`. If `dst_cols` is
        specified, the mapped concept name column will be named according to the
        corresponding name in `dst_cols`.

        Parameters
        ----------
        src_table: Subquery
            Source table with concept IDs.
        src_cols: str or list of str
            Column name(s) to consider as concept IDs for mapping.
        dst_cols: str or list of str, optional
            Column name(s) to assign for the mapped concept name columns.

        Returns
        -------
        Subquery
            Query with mapped columns from concept table.

        """
        concept_table = self.get_table(self.schema_name, "concept")
        src_cols = to_list(src_cols)
        if dst_cols:
            dst_cols = to_list(dst_cols)
            if len(dst_cols) != len(src_cols):
                raise ValueError("dst_cols must be same length as src_cols")

        for i, col in enumerate(src_cols):
            if ID not in col:
                raise ValueError("Specified column not a concept ID column!")
            src_table = qo.Join(
                concept_table,
                on=(col, CONCEPT_ID),
                join_table_cols=[CONCEPT_NAME],
                isouter=True,
            )(src_table)
            if dst_cols:
                dst_col_name = dst_cols[i]
            else:
                dst_col_name = col.replace(ID, NAME)
            src_table = qo.Rename({CONCEPT_NAME: dst_col_name})(src_table)

        return src_table

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
        care_site_table = self.get_table(self.schema_name, "care_site")
        source_table = qo.Join(
            care_site_table,
            on=CARE_SITE_ID,
            join_table_cols=[CARE_SITE_NAME, CARE_SITE_SOURCE_VALUE],
            isouter=True,
        )(source_table)

        return source_table

    def visit_occurrence(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query OMOP visit_occurrence table.

        Parameters
        ----------
        join: cyclops.query.ops.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(self.schema_name, "visit_occurrence")
        table = self.map_concept_ids_to_name(
            table,
            [
                "visit_concept_id",
                "visit_type_concept_id",
            ],
        )
        table = self._map_care_site_id(table)

        return QueryInterface(self.db, table, join=join, ops=ops)

    def visit_detail(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query OMOP visit_detail table.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(self.schema_name, "visit_detail")
        table = self.map_concept_ids_to_name(
            table, ["visit_detail_concept_id", "visit_detail_type_concept_id"]
        )

        return QueryInterface(self.db, table, join=join, ops=ops)

    def person(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query OMOP person table.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(self.schema_name, "person")
        table = self.map_concept_ids_to_name(
            table, ["gender_concept_id", "race_concept_id", "ethnicity_concept_id"]
        )

        return QueryInterface(self.db, table, join=join, ops=ops)

    def observation(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query OMOP observation table.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(self.schema_name, "observation")
        table = self.map_concept_ids_to_name(
            table, [OBSERVATION_CONCEPT_ID, OBSERVATION_TYPE_CONCEPT_ID]
        )

        return QueryInterface(self.db, table, join=join, ops=ops)

    def measurement(
        self,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Query OMOP measurement table.

        Parameters
        ----------
        join: qo.JoinArgs, optional
            Join arguments.
        ops: qo.Sequential, optional
            Additional operations to perform on query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            Constructed query, wrapped in an interface object.

        """
        table = self.get_table(self.schema_name, "measurement")
        # Cast value_as_concept_id to int.
        table = qo.Cast([VALUE_AS_CONCEPT_ID], "int")(table)
        table = self.map_concept_ids_to_name(
            table,
            [MEASUREMENT_CONCEPT_ID, MEASUREMENT_TYPE_CONCEPT_ID, UNIT_CONCEPT_ID],
        )

        return QueryInterface(self.db, table, join=join, ops=ops)
