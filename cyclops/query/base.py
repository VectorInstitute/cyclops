"""Base querier class."""
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from hydra import compose, initialize
from omegaconf import OmegaConf
from sqlalchemy import MetaData
from sqlalchemy.sql.selectable import Subquery

from cyclops.query import ops as qo
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.orm import Database
from cyclops.query.util import (
    DBSchema,
    TableTypes,
    _to_subquery,
    get_attr_name,
    table_params_to_type,
)
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def _create_get_table_lambdafn(schema_name: str, table_name: str) -> Callable[..., Any]:
    """Create a lambda function to access a table.

    Parameters
    ----------
    schema_name: str
        The schema name.
    table_name: str
        The table name.

    Returns
    -------
    Callable
        The lambda function.

    """
    return lambda db: getattr(getattr(db, schema_name), table_name)


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
    db: cyclops.query.orm.Database
        ORM Database used to run queries.

    Notes
    -----
    This class is intended to be subclassed to provide methods for querying tables in
    the database. This class automatically creates methods for querying tables in the
    database. The methods are named after the schema and table name, i.e.
    `self.schema_name.table_name()`. The methods are created when the class is
    instantiated. The subclass can provide custom methods for querying tables in the
    database which can build on the methods created by this class.

    """

    def __init__(
        self,
        **config_overrides: Dict[str, Any],
    ) -> None:
        """Initialize.

        Parameters
        ----------
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

        self.db = Database(config)
        if not self.db.is_connected:
            LOGGER.error("Database is not connected, cannot run queries.")
            return
        self._setup_table_methods()

    def list_schemas(self) -> List[str]:
        """List schemas in the database to query.

        Returns
        -------
        List[str]
            List of schema names.

        """
        return list(self.db.inspector.get_schema_names())

    def list_tables(self) -> List[str]:
        """List table methods that can be queried using the database.

        Returns
        -------
        List[str]
            List of table names.

        """
        return self.db.list_tables()

    def list_columns(self, schema_name: str, table_name: str) -> List[str]:
        """List columns in a table.

        Parameters
        ----------
        schema_name: str
            Name of schema in the database.
        table_name: str
            Name of GEMINI table.

        Returns
        -------
        List[str]
            List of column names.

        """
        return list(
            getattr(getattr(self.db, schema_name), table_name).data.columns.keys()
        )

    def list_custom_tables(self) -> List[str]:
        """List custom tables methods provided by the dataset API.

        Returns
        -------
        List[str]
            List of custom table names.

        """
        method_list = dir(self)
        custom_tables = []
        for method in method_list:
            if (
                not method.startswith(  # pylint: disable=too-many-boolean-expressions
                    "__"
                )
                and not method.startswith("_")
                and method not in self.list_schemas()
                and not method.startswith("list_")
                and not method.startswith("get_")
                and method not in ["db"]
            ):
                custom_tables.append(method)

        return custom_tables

    @table_params_to_type(Subquery)
    def get_interface(
        self,
        table: TableTypes,
        ops: Optional[qo.Sequential] = None,
        process_fn: Optional[Callable[..., Any]] = None,
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
            return QueryInterface(self.db, table, ops=ops)

        return QueryInterfaceProcessed(self.db, table, ops=ops, process_fn=process_fn)

    def get_table(
        self, schema_name: str, table_name: str, cast_timestamp_cols: bool = True
    ) -> Subquery:
        """Get a table and possibly map columns to have standard names.

        Standardizing column names allows for columns to be
        recognized in downstream processing.

        Parameters
        ----------
        schema_name: str
            Name of schema in the database.
        table_name: str
            Name of GEMINI table.
        cast_timestamp_cols: bool
            Whether to cast timestamp columns to datetime.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Table with mapped columns.

        """
        table = _create_get_table_lambdafn(schema_name, table_name)(self.db).data

        if cast_timestamp_cols:
            table = _cast_timestamp_cols(table)

        return _to_subquery(table)

    def _template_table_method(
        self,
        schema_name: str,
        table_name: str,
        join: Optional[qo.JoinArgs] = None,
        ops: Optional[qo.Sequential] = None,
    ) -> QueryInterface:
        """Template method for table methods.

        Parameters
        ----------
        schema_name: str
            Name of schema in the database.
        table_name: str
            Name of table in the database.
        join: cyclops.query.ops.JoinArgs
            Join arguments.
        ops: cyclops.query.ops.Sequential or cyclops.query.ops.QueryOp, optional
            Operations to perform on the query.

        Returns
        -------
        cyclops.query.interface.QueryInterface
            A query interface object.

        """
        table = getattr(getattr(self.db, schema_name), table_name).data
        table = _to_subquery(table)

        return QueryInterface(self.db, table, join=join, ops=ops)

    def _setup_table_methods(self) -> None:
        """Add table methods.

        This method adds methods to the querier class that allow querying of tables in
        the database. The methods are named after the table names.

        """
        schemas = self.list_schemas()
        meta: Dict[str, MetaData] = {}
        for schema_name in schemas:
            metadata = MetaData(schema=schema_name)
            metadata.reflect(bind=self.db.engine)
            meta[schema_name] = metadata
            schema = DBSchema(schema_name, meta[schema_name])
            for table_name in meta[schema_name].tables:
                setattr(
                    schema,
                    get_attr_name(table_name),
                    partial(
                        self._template_table_method,
                        schema_name=schema_name,
                        table_name=get_attr_name(table_name),
                    ),
                )
            setattr(self, schema_name, schema)
