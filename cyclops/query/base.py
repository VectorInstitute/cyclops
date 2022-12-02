"""Base querier class."""

import logging
from typing import Callable, Dict, List, Optional, Union

from hydra import compose, initialize
from omegaconf import OmegaConf
from sqlalchemy.sql.selectable import Subquery

from cyclops.query import process as qp
from cyclops.query.interface import QueryInterface, QueryInterfaceProcessed
from cyclops.query.orm import Database
from cyclops.query.util import TableTypes, _to_subquery, table_params_to_type
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class DatasetQuerier:
    """Base class to query EHR datasets.

    Attributes
    ----------
    _db: cyclops.query.orm.Database
        ORM Database used to run queries.
    _table_map: Dict
        A dictionary mapping table names to table objects in the DB.
    _column_map: Dict
        A dictionary mapping column names from the database to map to output
        in a consistent format.

    """

    def __init__(
        self, table_map: Dict, column_map: Dict, config_overrides: Optional[List] = None
    ) -> None:
        """Initialize.

        Parameters
        ----------
        config_overrides: list, optional
            List of override configuration parameters.
        table_map: Dict
            A dictionary mapping table names to table objects in the DB.
        column_map: Dict
            A dictionary mapping column names from the database to map to output
            in a consistent format.

        """
        if not config_overrides:
            config_overrides = []
        with initialize(
            version_base=None, config_path="configs", job_name="DatasetQuerier"
        ):
            config = compose(config_name="config", overrides=config_overrides)
            LOGGER.debug(OmegaConf.to_yaml(config))

        self._db = Database(config)
        self._table_map = table_map
        self._column_map = column_map

    @table_params_to_type(Subquery)
    def get_interface(
        self,
        table: TableTypes,
        process_fn: Optional[Callable] = None,
    ) -> Union[QueryInterface, QueryInterfaceProcessed]:
        """Get a query interface for a GEMINI table.

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
            return QueryInterface(self._db, table)

        return QueryInterfaceProcessed(self._db, table, process_fn)

    def get_table(self, table_name: str, rename: bool = True) -> Subquery:
        """Get a table and possibly map columns to have standard names.

        Standardizing column names allows for for columns to be
        recognized in downstream processing.

        Parameters
        ----------
        table_name: str
            Name of GEMINI table.
        rename: bool, optional
            Whether to map the column names

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Table with mapped columns.

        """
        if table_name not in self._table_map:
            raise ValueError(f"{table_name} not a recognised table.")

        table = self._table_map[table_name](self._db).data

        if rename:
            table = qp.Rename(self._column_map, check_exists=False)(table)

        return _to_subquery(table)
