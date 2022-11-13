"""A query interface class to wrap database objects and queries."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Optional, Union

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
    data: pandas.DataFrame
        Data returned from executing the query, as Pandas DataFrame.
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.

    """

    database: Database
    query: TableTypes
    data: Optional[pd.DataFrame] = None
    _run_args: Dict = field(default_factory=dict)

    def run(
        self,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit: int, optional
            No. of rows to limit the query return.

        Returns
        -------
        pandas.DataFrame
            Query result dataframe.

        """
        # Only re-run when new run arguments are given
        if self.data is None or not self._run_args == locals():
            self._run_args = locals()
            self.data = self.database.run_query(self.query, limit=limit)

        return self.data

    def save(self, path: str, file_format: str = "parquet") -> str:
        """Save the query.

        Parameters
        ----------
        save_path: str
            Path where the file will be saved.
        file_format: str
            File format of the file to save.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        # If the query was already run
        if self.data is not None:
            path = save_dataframe(self.data, path, file_format=file_format)
            return path

        # Save without running
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

    def run_in_grouped_batches(
        self, id_col: str, batch_size: int
    ) -> Generator[pd.DataFrame, None, None]:
        """Run the query in batches with complete sets of sample IDs.

        Queries are sorted and grouped such that the rows for a given sample ID are kept
        together in a single batch.

        Parameters
        ----------
        id_col: str
            Name of the sample ID column by which to batch.
        batch_size: int
            Approximate batch size before rearranging based on sample IDs.

        Yields
        ------
        pandas.DataFrame
            A query batch with complete sets of sample IDs.

        """
        generator = self.database.run_id_batched_query(self.query, id_col, batch_size)
        while True:
            try:
                yield next(generator)
            except StopIteration:
                return

    def save_in_grouped_batches(
        self,
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
        self.database.save_id_batched_query(
            self.query, dir_path, id_col, batch_size, file_format=file_format
        )


@dataclass
class QueryInterfaceProcessed:
    """An interface dataclass to wrap queries, and run them with post-processing.

    An similar dataclass to QueryInterface, where custom post-processing
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

    Attributes
    ----------
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.
    data: pandas.DataFrame
        Data returned from executing the query, as Pandas DataFrame.

    """

    database: Database
    _query: TableTypes
    process_fn: Callable
    data: Union[pd.DataFrame, None] = None
    _run_args: Dict = field(default_factory=dict)

    def run(
        self,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit: int, optional
            No. of rows to limit the query return.

        Returns
        -------
        pandas.DataFrame
            Query result dataframe.

        """
        if self.data is None or not self._run_args == locals():
            self._run_args = locals()
            self.data = self.database.run_query(self._query, limit=limit)

            LOGGER.info(
                "Applying post-processing fn %s to query output",
                self.process_fn.__name__,
            )
            return self.process_fn(self.data)

        return self.data

    def save(self, path: str, file_format: str = "parquet") -> str:
        """Save the processed query.

        Parameters
        ----------
        save_path: str
            Path where the file will be saved.
        file_format: str
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
