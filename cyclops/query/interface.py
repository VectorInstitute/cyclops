"""A query interface class to wrap database objects and queries."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops.orm import Database
from cyclops.processors.column_names import RECOGNISED_QUERY_COLUMNS
from cyclops.query.util import filter_attributes
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class QueryInterface:
    """An interface dataclass to actually wrap queries, and run them.

    Parameters
    ----------
    database: cyclops.orm.Database
        Database object to create ORM, and query data.
    query: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery
        The query.
    data: pandas.DataFrame
        Data returned from executing the query, as Pandas DataFrame.

    """

    database: Database
    query: Union[Select, Subquery]
    data: Union[pd.DataFrame, None] = None
    _run_args: Optional[Dict] = {}

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
        if self.data is None and not self._run_args == locals():
            self._run_args = locals()
            self.data = self.database.run_query(self.query, limit=limit)

        return self.data

    def __repr__(self) -> str:
        """Print SQL when printing the QueryInterface object."""
        return str(self.query)

    def save(self, folder_path: str, file_name: str) -> None:
        """Save queried data in Parquet format.

        Parameters
        ----------
        folder_path: str
            Path to directory where the file can be saved.
        file_name: str
            Name of file. Extension will be .gzip.

        """
        save_path = os.path.join(folder_path, file_name + ".gzip")
        if isinstance(self.data, pd.DataFrame):
            LOGGER.info("Saving queried data to %s", save_path)
            self.data.to_parquet(save_path)
        elif self.data is None:
            LOGGER.warning("Query not run, no data to save!")
