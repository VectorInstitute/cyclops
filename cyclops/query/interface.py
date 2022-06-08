"""A query interface class to wrap database objects and queries."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.orm import Database
from cyclops.query.util import TableTypes
from cyclops.utils.file import save_dataframe
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


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

    def save(self, path: str, file_format: str = "parquet") -> None:
        """Save the query.

        Parameters
        ----------
        save_path: str
            Path where the file will be saved.
        file_format: str
            File format of the file to save.

        """
        # If the query was already run
        if self.data is not None:
            save_dataframe(self.data, path, file_format=file_format)
            return

        # Save without running
        if file_format == "csv":
            self.database.save_query_to_csv(self.query, path)
        elif file_format == "parquet":
            self.database.save_query_to_parquet(self.query, path)
        else:
            raise ValueError("Invalid file format specified.")

    def clear_data(self) -> None:
        """Clear data container.

        Sets the data attribute to None, thus clearing the dataframe contained.

        """
        self.data = None


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
    data: pandas.DataFrame
        Data returned from executing the query, as Pandas DataFrame.
    process_fn_kwargs: dict
        Keyword arguments for post-processing function.
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.

    """

    database: Database
    _query: TableTypes
    process_fn: Callable
    data: Union[pd.DataFrame, None] = None
    process_fn_kwargs: Dict = field(default_factory=dict)
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
            if self.process_fn:
                LOGGER.info(
                    "Applying post-processing fn %s to query output",
                    self.process_fn.__name__,
                )
                self.data = self.process_fn(self.data, **self.process_fn_kwargs)

        return self.data

    def save(self, path: str, file_format: str = "parquet") -> None:
        """Save the processed query.

        Parameters
        ----------
        save_path: str
            Path where the file will be saved.
        file_format: str
            File format of the file to save.

        """
        # The query must be run in order to be processed.
        if self.data is None:
            self.run()

        save_dataframe(self.data, path, file_format=file_format)

    def clear_data(self) -> None:
        """Clear data container.

        Sets the data attribute to None, thus clearing the dataframe contained.

        """
        self.data = None
