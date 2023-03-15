"""A query interface class to wrap database objects and queries."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Union

import dask.dataframe as dd
import pandas as pd

import cyclops.query.ops as qo
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
    join: cyclops.query.ops.JoinArgs, optional
        Join arguments to join the query with another table.
    ops: cyclops.query.ops.Sequential or cyclops.query.ops.QueryOp, optional
        Operations to perform on the query.
    _data: pandas.DataFrame or dask.DataFrame
        Data returned from executing the query, as Pandas DataFrame.
    _run_args: dict Private dictionary attribute to keep track of arguments
        passed to run() method.

    Notes
    -----
    After initialization, the query, join operation and chaining of provided operations,
    are automatically done and the query attribute is updated.

    """

    database: Database
    query: TableTypes
    join: Optional[qo.JoinArgs] = None
    ops: Optional[Union[qo.QueryOp, qo.Sequential]] = None
    _data: Optional[Union[pd.DataFrame, dd.core.DataFrame]] = None
    _run_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post init method to chain operations with original query."""
        if self.join is not None:
            self.query = qo.Join(**asdict(self.join))(self.query)
        if self.ops is not None:
            self.query = self.ops(self.query)

    @property
    def data(self) -> Optional[Union[pd.DataFrame, dd.core.DataFrame]]:
        """Get data."""
        return self._data

    def run(
        self,
        limit: Optional[int] = None,
        backend: Literal["pandas", "dask"] = "pandas",
        index_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> Union[pd.DataFrame, dd.core.DataFrame]:
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
        if self._data is None or not self._run_args == locals():
            self._run_args = locals()
            self._data = self.database.run_query(
                self.query,
                limit=limit,
                backend=backend,
                index_col=index_col,
                n_partitions=n_partitions,
            )

        return self._data

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
        if self._data is not None:
            path = save_dataframe(self._data, path, file_format=file_format)
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
        self._data = None


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
    join: cyclops.query.ops.JoinArgs, optional
        Join arguments to join the query with another table.
    ops: cyclops.query.ops.Sequential or cyclops.query.ops.QueryOp, optional
        Operations to perform on the query.
    _data: pandas.DataFrame or dask.DataFrame
        Data returned from executing the query, as Pandas DataFrame.
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.

    Notes
    -----
    After initialization, the query, join operation and chaining of provided operations,
    are automatically done and the query attribute is updated.

    """

    database: Database
    _query: TableTypes
    process_fn: Callable[..., Any]
    join: Optional[qo.JoinArgs] = None
    ops: Optional[Union[qo.QueryOp, qo.Sequential]] = None
    _data: Optional[Union[pd.DataFrame, dd.core.DataFrame, None]] = None
    _run_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post init method to chain operations with original query."""
        if self.join is not None:
            self._query = qo.Join(**asdict(self.join))(self._query)
        if self.ops is not None:
            self._query = self.ops(self._query)

    def run(
        self,
        limit: Optional[int] = None,
        backend: Literal["pandas", "dask"] = "pandas",
        index_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> Union[pd.DataFrame, dd.core.DataFrame]:
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
        if self._data is None or not self._run_args == locals():
            self._run_args = locals()
            self._data = self.database.run_query(
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
            return self.process_fn(self._data)

        return self._data

    @property
    def data(self) -> Optional[Union[pd.DataFrame, dd.core.DataFrame]]:
        """Get data."""
        return self._data

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
        if self._data is None:
            self.run()
        path = save_dataframe(self._data, path, file_format=file_format)

        return path

    def clear_data(self) -> None:
        """Clear data container.

        Sets the data attribute to None, thus clearing the dataframe contained.

        """
        self._data = None
