"""A query interface class to wrap database objects and queries."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops.orm import Database
from cyclops.processors.column_names import RECOGNISED_QUERY_COLUMNS
from cyclops.query.util import filter_attributes
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
    query: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery
        The query.
    data: pandas.DataFrame
        Data returned from executing the query, as Pandas DataFrame.
    _run_args: dict
        Private dictionary attribute to keep track of arguments
        passed to run() method.

    """

    database: Database
    query: Union[Select, Subquery]
    data: Union[pd.DataFrame, None] = None
    _run_args: Dict = field(default_factory=dict)

    def run(
        self,
        limit: Optional[int] = None,
        filter_columns: Optional[Union[str, List[str]]] = None,
        filter_recognised: bool = False,
    ) -> pd.DataFrame:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit: int, optional
            No. of rows to limit the query return.
        filter_columns: str or list of str, optional
            Filters specified columns from returned dataframe, if they are present.
        filter_recognised: bool, optional
            Filter columns that are recognised by the processor. Useful to avoid
            increased RAM usage when running entire pipelines.

        Returns
        -------
        pandas.DataFrame
            Query result dataframe.

        Raises
        ------
        ValueError
            When filter_columns and filter_recognised are both mistakenly used,
            an error is raised.

        """
        if filter_columns and filter_recognised:
            raise ValueError(
                "Both filter_recognised and filter_columns cannot be used!"
            )
        if filter_columns:
            self.query = filter_attributes(self.query, filter_columns)
        if filter_recognised:
            self.query = filter_attributes(self.query, RECOGNISED_QUERY_COLUMNS)
        if self.data is None or not self._run_args == locals():
            self._run_args = locals()
            self.data = self.database.run_query(self.query, limit=limit)

        return self.data

    def save(self, folder_path: str, file_name: str) -> None:
        """Save queried data in Parquet format.

        Parameters
        ----------
        folder_path: str
            Path to directory where the file can be saved.
        file_name: str
            Name of file. Extension will be .gzip.

        """
        save_dataframe(self.data, folder_path, file_name, prefix="query")

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
    _query: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery
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
    _query: Union[Select, Subquery]
    process_fn: Callable
    data: Union[pd.DataFrame, None] = None
    process_fn_kwargs: Dict = field(default_factory=dict)
    _run_args: Dict = field(default_factory=dict)

    def run(
        self,
        limit: Optional[int] = None,
        filter_columns: Optional[Union[str, List[str]]] = None,
        filter_recognised: bool = False,
    ) -> pd.DataFrame:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit: int, optional
            No. of rows to limit the query return.
        filter_columns: str or list of str, optional
            Filters specified columns from returned dataframe, if they are present.
        filter_recognised: bool, optional
            Filter columns that are recognised by the processor. Useful to avoid
            increased RAM usage when running entire pipelines.

        Returns
        -------
        pandas.DataFrame
            Query result dataframe.

        Raises
        ------
        ValueError
            When filter_columns and filter_recognised are both mistakenly used,
            an error is raised.

        """
        if filter_columns and filter_recognised:
            raise ValueError(
                "Both filter_recognised and filter_columns cannot be used!"
            )
        if filter_columns:
            self._query = filter_attributes(self._query, filter_columns)
        if filter_recognised:
            self._query = filter_attributes(self._query, RECOGNISED_QUERY_COLUMNS)
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

    def save(self, folder_path: str, file_name: str) -> None:
        """Save queried data in Parquet format.

        Parameters
        ----------
        folder_path: str
            Path to directory where the file can be saved.
        file_name: str
            Name of file. Extension will be .gzip.

        """
        save_dataframe(self.data, folder_path, file_name, prefix="query")

    def clear_data(self) -> None:
        """Clear data container.

        Sets the data attribute to None, thus clearing the dataframe contained.

        """
        self.data = None
