"""Object Relational Mapper (ORM) using sqlalchemy."""

import csv
import logging
import os
import socket
from typing import Generator, List, Optional, Union

import pandas as pd
import pyarrow.csv as pv
import pyarrow.parquet as pq
from omegaconf import DictConfig
from sqlalchemy import MetaData, and_, create_engine, func, inspect, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops.query.util import (
    DBSchema,
    DBTable,
    TableTypes,
    get_column,
    table_params_to_type,
)
from cyclops.utils.file import (
    exchange_extension,
    join,
    process_dir_save_path,
    process_file_save_path,
    save_dataframe,
)
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
        config: omegaconf.dictconfig.DictConfig
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
        query: TableTypes,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run query.

        Parameters
        ----------
        query: cyclops.query.util.TableTypes
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
            query = query.limit(limit)  # type: ignore

        # Run the query and return the results
        with self.session.connection():
            data = pd.read_sql_query(query, self.engine)

        LOGGER.info("Query returned successfully!")
        return data

    @time_function
    def run_sql_string(
        self,
        query: str,
    ) -> pd.DataFrame:
        """Run query from SQL raw SQL string.

        Parameters
        ----------
        query: str
            Raw SQL query string.

        Returns
        -------
        pd.DataFrame
            Extracted data from query.

        """
        # Run the query and return the results
        with self.session.connection():
            data = pd.read_sql_query(query, self.engine)

        LOGGER.info("Query returned successfully!")
        return data

    @time_function
    @table_params_to_type(Select)
    def save_query_to_csv(self, query: TableTypes, path: str) -> str:
        """Save query in a .csv format.

        Parameters
        ----------
        query: cyclops.query.util.TableTypes
            Query to save.
        path:
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
        query: cyclops.query.util.TableTypes
            Query to save.
        path:
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

    def query_batch_conditions(
        self,
        query: TableTypes,
        id_col: str,
        batch_size: int,
    ) -> List[Union[BinaryExpression, BooleanClauseList]]:
        """Return a list of WHERE conditions to segment a query into batches.

        Batches are created via SQL windowing, based on segmenting the values in a
        given column, such as an ID column, into intervals.

        Requires a database that supports window functions.

        Parameters
        ----------
        column: sqlalchemy.sql.schema.Column
            The column over which to create the interval ranges and conditions.
        window_size: int
            The range of the interval to create.

        Returns
        -------
        list of sqlalchemy.sql.elements.BinaryExpression or
        sqlalchemy.sql.elements.BooleanClauseList
            The window conditions on which to filter.

        """

        def compute_query_dividers(query, id_col, maximum):
            # Compute the row count for each unique value
            col = get_column(query, id_col)
            table = select(col, func.count(col).label("count")).group_by(col)
            count_data = self.run_query(table)

            # Check that all values can actually fit into the maximum batch size
            max_count = count_data["count"].max()
            if maximum < max_count:
                raise ValueError(f"Maximum must be at least {max_count}.")

            # Sort and create a cumulative sum of row counts
            count_data = count_data.sort_values(id_col)
            count_data["cumsum"] = count_data["count"].cumsum()

            # Create query dividers
            last_sum = 0

            if len(count_data) == 0:
                raise ValueError("Query is empty. Cannot return batched results.")

            dividers = [int(count_data[id_col].iloc[0])]
            for i, cumsum in enumerate(count_data["cumsum"].values[1:]):
                # If adding the next value will put the sum over the max,
                # then add another divider on the previous value
                if cumsum - last_sum > maximum:
                    dividers.append(int(count_data[id_col].iloc[i]))
                    last_sum = count_data["cumsum"].iloc[i]

            return dividers

        def range_condition(start_id, end_id):
            if end_id:
                return and_(column >= start_id, column < end_id)

            return column >= start_id

        # Create interval dividers
        dividers = compute_query_dividers(query, id_col, batch_size)
        # print("dividers", dividers)

        # Create filtering conditions
        column = get_column(query, id_col)
        conditions = []
        while dividers:
            # Create interval ranges
            start = dividers.pop(0)
            if dividers:
                end = dividers[0]
            else:
                end = None

            # Create condition
            conditions.append(range_condition(start, end))

        return conditions

    @table_params_to_type(Subquery)
    def run_id_batched_query(
        self,
        query: TableTypes,
        id_col: str,
        batch_size: int,
    ) -> Generator[pd.DataFrame, None, None]:
        """Generate query batches with complete sets of IDs in a batch.

        Queries are sorted and grouped such that the rows for a given sample ID are kept
        together in a single batch.

        Parameters
        ----------
        query: cyclops.query.util.TableTypes
            Query to run.
        window_size: int
            Window size used to batch queries over the ID column.
        id_col: str
            Name of the sample ID column by which to batch.

        Yields
        ------
        pandas.DataFrame
            A query batch with complete sets of sample IDs.

        """
        if "limit" in str(query).lower():
            raise NotImplementedError(
                "Currently not supporting batching for queries with a LIMIT."
            )

        conditions = self.query_batch_conditions(query, id_col, batch_size)
        sess_query = self.session.query(query)

        # Opportunity for easy multi-processing/parallelization here!
        for condition in conditions:
            run = (sess_query.filter(condition)).subquery()
            yield pd.read_sql_query(run, self.engine)

    @time_function
    def save_id_batched_query(  # pylint: disable=too-many-arguments
        self,
        query: TableTypes,
        dir_path: str,
        id_col: str,
        batch_size: int,
        file_format: str = "parquet",
    ) -> None:
        """Save a query in different batches, keeping same sample IDs together.

        Queries are sorted and grouped such that the rows for a given sample ID are kept
        together in a single batch.

        Parameters
        ----------
        query: cyclops.query.util.TableTypes
            Query to run.
        dir_path: str
            Path to directory in which to save the batches.
        batch_size: int
            Approximate batch size before rearranging based on sample IDs.
        id_col: str
            Name of the sample ID column by which to batch.
        file_format: str
            File format of the DataFrame to save.

        """
        dir_path = process_dir_save_path(dir_path)
        generator = self.run_id_batched_query(query, id_col, batch_size)
        for i, batch in enumerate(generator):
            save_dataframe(
                batch,
                join(dir_path, "batch_" + f"{i:04d}"),
                file_format=file_format,
            )
