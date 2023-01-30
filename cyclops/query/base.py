"""Base querier class."""

import logging
from functools import partial
from typing import Callable, List, Mapping, Optional, Union

from hydra import compose, initialize
from omegaconf import OmegaConf
from sqlalchemy.sql.selectable import Subquery

from cyclops.query import ops as qo
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.orm import Database
from cyclops.query.util import TableTypes, _to_subquery, table_params_to_type
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def _cast_timestamp_cols(table: Subquery) -> Subquery:
    """Cast timestamp columns to datetime.

    Parameters
    ----------
    table: sqlalchemy.sql.selectable.Subquery
        Table to cast.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Table with cast columns.

    """
    cols_to_cast = []
    for col in table.columns:
        if str(col.type) == "TIMESTAMP":
            cols_to_cast.append(col.name)
    if cols_to_cast:
        table = qo.Cast(cols_to_cast, "timestamp")(table)

    return table


class DatasetQuerier:
    """Base class to query EHR datasets.

    Attributes
    ----------
    _db: cyclops.query.orm.Database
        ORM Database used to run queries.
    _table_map: Mapping
        A dictionary mapping table names to table objects in the DB.

    Notes
    -----
    This class is intended to be subclassed to provide methods for querying tables in
    the database. The subclass accepts a table map as an argument, which is a mapping
    from table names to table objects. By default, the methods are named after the
    table names. The subclass can override the table methods by defining a method with
    the same name as the table. The subclass can also add additional methods.

    """

    def __init__(self, table_map: Mapping, **config_overrides) -> None:
        """Initialize.

        Parameters
        ----------
        table_map: Mapping
            A dictionary mapping table names to table objects in the DB.
        **config_overrides
             Override configuration parameters, specified as kwargs.

        """
        overrides = []
        if config_overrides:
            for key, value in config_overrides.items():
                overrides.append(f"{key}={value}")
        with initialize(
            version_base=None, config_path="configs", job_name="DatasetQuerier"
        ):
            config = compose(config_name="config", overrides=overrides)
            LOGGER.debug(OmegaConf.to_yaml(config))

        self._db = Database(config)
        self._table_map = table_map
        self._setup_table_methods()

    def list_tables(self) -> List[str]:
        """List table methods that can be queried using the database.

        Returns
        -------
        List[str]
            List of table names.

        """
        return list(self._table_map.keys())

    @table_params_to_type(Subquery)
    def get_interface(
        self,
        table: TableTypes,
        ops: Optional[qo.Sequential] = None,
        process_fn: Optional[Callable] = None,
    ) -> Union[QueryInterface, QueryInterfaceProcessed]:
        """Get a query interface for a GEMINI table.

        Parameters
        ----------
        table: cyclops.query.util.TableTypes
            Table to wrap in the interface.
        ops: cyclops.query.ops.Sequential
            Operations to perform on the query.
        process_fn
            Process function to apply on the Pandas DataFrame returned from the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface or
        cyclops.query.interface.QueryInterfaceProcessed
            A query interface using the GEMINI database object.

        """
        if process_fn is None:
            return QueryInterface(self._db, table, ops=ops)

        return QueryInterfaceProcessed(self._db, table, ops=ops, process_fn=process_fn)

    def get_table(self, table_name: str, cast_timestamp_cols: bool = True) -> Subquery:
        """Get a table and possibly map columns to have standard names.

        Standardizing column names allows for for columns to be
        recognized in downstream processing.

        Parameters
        ----------
        table_name: str
            Name of GEMINI table.
        cast_timestamp_cols: bool
            Whether to cast timestamp columns to datetime.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Table with mapped columns.

        """
        if table_name not in self._table_map:
            raise ValueError(f"{table_name} not a recognised table.")

        table = self._table_map[table_name](self._db).data

        if cast_timestamp_cols:
            table = _cast_timestamp_cols(table)

        return _to_subquery(table)

    def _template_table_method(
        self,
        table_name: str,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Template method for table methods.

        Parameters
        ----------
        table_name: str
            Name of table in the database.
        join: cyclops.query.ops.JoinArgs
            Join arguments.
        ops: cyclops.query.ops.Sequential
            Operations to perform on the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            A query interface object.

        """
        table = self.get_table(table_name)

        return QueryInterface(self._db, table, join=join, ops=ops)

    def _setup_table_methods(self) -> None:
        """Add table methods.

        This method adds methods to the querier class that allow querying of tables in
        the database. The methods are named after the table names.

        """
        for table_name in self._table_map:
            if not hasattr(self, table_name):
                setattr(
                    self,
                    table_name,
                    partial(self._template_table_method, table_name=table_name),
                )
