"""Object Relational Mapper (ORM) using sqlalchemy."""

import argparse
import logging
import socket
from typing import Optional, Union

import pandas as pd
from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.schema import Table
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops.query.util import DBMetaclass, DBSchema, DBTable, query_params_to_type
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


SOCKET_CONNECTION_TIMEOUT = 5


def _get_db_url(  # pylint: disable=too-many-arguments
    dbms: str, user: str, pwd: str, host: str, port: str, database: str
) -> str:
    """Combine to make Database URL string."""
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{database}"


def _get_attr_name(name: str) -> str:
    """Get attribute name (second part of first.second)."""
    return name.split(".")[-1]


class Database(metaclass=DBMetaclass):  # pylint: disable=too-few-public-methods
    """Database class (singleton).

    Attributes
    ----------
    config: argparse.Namespace
        Configuration stored in object.
    engine: sqlalchemy.engine.base.Engine
        SQL extraction engine.
    inspector: sqlalchemy.engine.reflection.Inspector
        Module for schema inspection.
    session: sqlalchemy.orm.session.Session

    """

    def __init__(self, config: argparse.Namespace):
        """Instantiate.

        Parameters
        ----------
        config: argparse.Namespace
            Configuration stored in object.

        """
        self.config = config
        self.engine = None
        self.inspector = None

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SOCKET_CONNECTION_TIMEOUT)
        try:
            is_port_open = sock.connect_ex((config.host, config.port))
        except socket.gaierror:
            LOGGER.error("""Server name not known, cannot establish connection!""")
            return
        if is_port_open:
            LOGGER.error(
                """Valid server host but port seems open, check if server is up!"""
            )
            return
        self._create_engine()

    def _create_engine(self) -> None:
        """Attempt to create an engine, connect to DB."""
        self.engine = create_engine(
            _get_db_url(
                self.config.dbms,
                self.config.user,
                self.config.password,
                self.config.host,
                self.config.port,
                self.config.database,
            ),
        )
        self.inspector = inspect(self.engine)

        # Create a session for using ORM.
        session = sessionmaker(self.engine)
        session.configure(bind=self.engine)
        self.session = session()

        self._setup()
        LOGGER.info("Database setup, ready to run queries!")

    def _setup(self):
        """Prepare ORM DB."""
        meta: dict = {}
        schemas = self.inspector.get_schema_names()
        for schema_name in schemas:
            metadata = MetaData(schema=schema_name)
            metadata.reflect(bind=self.engine)
            meta[schema_name] = metadata
            schema = DBSchema(schema_name, meta[schema_name])
            for table_name in meta[schema_name].tables:
                table = DBTable(table_name, meta[schema_name].tables[table_name])
                for column in meta[schema_name].tables[table_name].columns:
                    setattr(table, column.name, column)
                if not isinstance(table.name, str):
                    table.name = str(table.name)
                setattr(schema, _get_attr_name(table.name), table)
            setattr(self, schema_name, schema)

    @time_function
    @query_params_to_type(Select)
    def run_query(
        self,
        query: Union[Select, Subquery, Table, DBTable],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run query.

        Parameters
        ----------
        query: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or DBTable
            Query to run.
        limit: Optional[int]
            Limit query result to limit.

        Returns
        -------
        pd.DataFrame
            Extracted data from query.

        """
        # Limit the results returned
        if limit is not None:
            query = query.limit(limit)

        # Run the query and return the results
        with self.session.connection():
            data = pd.read_sql_query(query, self.engine)

        LOGGER.info("Query returned successfully!")
        return data
