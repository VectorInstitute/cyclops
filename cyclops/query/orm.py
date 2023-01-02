"""Object Relational Mapper (ORM) using sqlalchemy."""

import csv
import logging
import os
import socket
from typing import Literal, Optional, Union

import dask.dataframe as dd
import pandas as pd
import pyarrow.csv as pv
import pyarrow.parquet as pq
from omegaconf import DictConfig
from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.selectable import Select

from cyclops.query.util import DBSchema, DBTable, TableTypes, table_params_to_type
from cyclops.utils.file import exchange_extension, process_file_save_path
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


SOCKET_CONNECTION_TIMEOUT = 5


def _get_db_url(  # pylint: disable=too-many-arguments
    dbms: str, user: str, pwd: str, host: str, port: str, database: str
) -> str:
    """Combine to make Database URL string."""
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{database}"


def _get_attr_name(name: str) -> str:
    """Get attribute name (second part of first.second)."""
    return name.split(".")[-1]


class Database:
    """Database class.

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

    def __init__(self, config: DictConfig):
        """Instantiate.

        Parameters
        ----------
        config
            Path to directory with config file, for overrides.

        """
        self.config = config

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(SOCKET_CONNECTION_TIMEOUT)
        try:
            is_port_open = sock.connect_ex((self.config.host, self.config.port))
        except socket.gaierror:
            LOGGER.error("""Server name not known, cannot establish connection!""")
            return
        if is_port_open:
            LOGGER.error(
                """Valid server host but port seems open, check if server is up!"""
            )
            return

        self.engine = self._create_engine()
        self.session = self._create_session()
        self._setup()
        LOGGER.info("Database setup, ready to run queries!")

    def _create_engine(self) -> Engine:
        """Create an engine."""
        self.conn = _get_db_url(
            self.config.dbms,
            self.config.user,
            self.config.password,
            self.config.host,
            self.config.port,
            self.config.database,
        )
        engine = create_engine(
            _get_db_url(
                self.config.dbms,
                self.config.user,
                self.config.password,
                self.config.host,
                self.config.port,
                self.config.database,
            ),
        )
        return engine

    def _create_session(self) -> Session:
        """Create session."""
        self.inspector = inspect(self.engine)

        # Create a session for using ORM.
        session = sessionmaker(self.engine)
        session.configure(bind=self.engine)
        return session()

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
    @table_params_to_type(Select)
    def run_query(
        self,
        query: Union[TableTypes, str],
        limit: Optional[int] = None,
        backend: Literal["pandas", "dask"] = "pandas",
        index_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Run query.

        Parameters
        ----------
        query
            Query to run.
        limit
            Limit query result to limit.
        backend
            Backend computing framework to use, Pandas or Dask.
        index_col
            Column which becomes the index, and defines the partitioning.
            Should be a indexed column in the SQL server, and any orderable type.
        n_partitions
            Number of partitions. Check dask documentation for additional details.

        Returns
        -------
        pandas.DataFrame or dask.DataFrame
            Extracted data from query.

        """
        if isinstance(query, str) and limit is not None:
            raise ValueError(
                "Cannot use limit argument when running raw SQL string query!"
            )
        if backend == "pandas" and n_partitions is not None:
            raise ValueError("Partitions not applicable with Pandas backend, use Dask!")
        # Limit the results returned.
        if limit is not None:
            query = query.limit(limit)  # type: ignore

        # Run the query and return the results.
        with self.session.connection():
            if backend == "pandas":
                data = pd.read_sql_query(query, self.engine, index_col=index_col)
            elif backend == "dask":
                data = dd.read_sql_query(
                    query, self.conn, index_col=index_col, npartitions=n_partitions
                )
                data = data.reset_index(drop=False)
            else:
                raise ValueError("Invalid backend, can either be Pandas or Dask!")
        LOGGER.info("Query returned successfully!")

        return data

    @time_function
    @table_params_to_type(Select)
    def save_query_to_csv(self, query: TableTypes, path: str) -> str:
        """Save query in a .csv format.

        Parameters
        ----------
        query
            Query to save.
        path
            Save path.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        path = process_file_save_path(path, "csv")

        with self.session.connection():
            result = self.engine.execute(query)
            with open(path, "w", encoding="utf-8") as file_descriptor:
                outcsv = csv.writer(file_descriptor)
                outcsv.writerow(result.keys())
                outcsv.writerows(result)

        return path

    @time_function
    @table_params_to_type(Select)
    def save_query_to_parquet(self, query: TableTypes, path: str) -> str:
        """Save query in a .parquet format.

        Parameters
        ----------
        query
            Query to save.
        path
            Save path.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        path = process_file_save_path(path, "parquet")

        # Save to CSV, load with pyarrow, save to Parquet
        csv_path = exchange_extension(path, "csv")
        self.save_query_to_csv(query, csv_path)
        table = pv.read_csv(csv_path)
        os.remove(csv_path)
        pq.write_table(table, path)

        return path
