"""A query interface class to wrap database objects and queries."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Union

import dask.dataframe as dd
import pandas as pd

from cyclops.query.orm import Database
from cyclops.query.util import TableTypes
from cyclops.utils.file import save_dataframe
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


@dataclass
class QueryInterface:
    """An interface dataclass to wrap queries, and run them.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database object to create ORM, and query data.
    query: cyclops.query.util.TableTypes
        The query.
    data: pandas.DataFrame or dask.DataFrame
        Data returned from executing the query, as Pandas DataFrame.
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.

    """

    database: Database
    query: TableTypes
    data: Optional[Union[pd.DataFrame, dd.DataFrame]] = None
    _run_args: Dict = field(default_factory=dict)

    def run(
        self,
        limit: Optional[int] = None,
        backend: Literal["pandas", "dask"] = "pandas",
        index_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit
            No. of rows to limit the query return.
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
            Query result dataframe.

        """
        # Only re-run when new run arguments are given.
        if self.data is None or not self._run_args == locals():
            self._run_args = locals()
            self.data = self.database.run_query(
                self.query,
                limit=limit,
                backend=backend,
                index_col=index_col,
                n_partitions=n_partitions,
            )

        return self.data

    def save(
        self, path: str, file_format: Literal["parquet", "csv"] = "parquet"
    ) -> str:
        """Save the query.

        Parameters
        ----------
        save_path
            Path where the file will be saved.
        file_format
            File format of the file to save.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        # If the query was already run.
        if self.data is not None:
            path = save_dataframe(self.data, path, file_format=file_format)
            return path

        # Save without running.
        if file_format == "csv":
            path = self.database.save_query_to_csv(self.query, path)
        elif file_format == "parquet":
            path = self.database.save_query_to_parquet(self.query, path)
        else:
            raise ValueError("Invalid file format specified.")

        return path

    def clear_data(self) -> None:
        """Clear data container.

        Sets the data attribute to None, thus clearing the dataframe contained.

        """
        self.data = None


@dataclass
class QueryInterfaceProcessed:
    """An interface dataclass to wrap queries, and run them with post-processing.

    A similar dataclass to QueryInterface, where custom post-processing
    functions on the pandas dataframe returned from the query can be run. However,
    this prevents the query from being further used, and hence is declared as
    private attribute.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database object to create ORM, and query data.
    _query: cyclops.query.util.TableTypes
        The query.
    process_fn: Callable
        Process function to apply on the pandas dataframe returned from the query.
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.
    data: pandas.DataFrame or dask.DataFrame
        Data returned from executing the query, as Pandas DataFrame.

    """

    database: Database
    _query: TableTypes
    process_fn: Callable
    data: Optional[Union[pd.DataFrame, dd.DataFrame, None]] = None
    _run_args: Dict = field(default_factory=dict)

    def run(
        self,
        limit: Optional[int] = None,
        backend: Literal["pandas", "dask"] = "pandas",
        index_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit
            No. of rows to limit the query return.
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
            Query result dataframe.

        """
        # Only re-run when new run arguments are given.
        if self.data is None or not self._run_args == locals():
            self._run_args = locals()
            self.data = self.database.run_query(
                self._query,
                limit=limit,
                backend=backend,
                index_col=index_col,
                n_partitions=n_partitions,
            )

            LOGGER.info(
                "Applying post-processing fn %s to query output",
                self.process_fn.__name__,
            )
            return self.process_fn(self.data)

        return self.data

    def save(
        self, path: str, file_format: Literal["parquet", "csv"] = "parquet"
    ) -> str:
        """Save the processed query.

        Parameters
        ----------
        save_path
            Path where the file will be saved.
        file_format
            File format of the file to save.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        # The query must be run in order to be processed.
        if self.data is None:
            self.run()
        path = save_dataframe(self.data, path, file_format=file_format)

        return path

    def clear_data(self) -> None:
        """Clear data container.

        Sets the data attribute to None, thus clearing the dataframe contained.

        """
        self.data = None
