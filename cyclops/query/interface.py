"""A query interface class to wrap database objects and queries."""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops.orm import Database
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class QueryInterface:
    """An interface dataclass to actually wrap queries, and run them.

    Parameters
    ----------
    _db: cyclops.orm.Database
        Database object to create ORM, and query data.
    query: sqlalchemy.sql.selectable.Select or
    sqlalchemy.sql.selectable.Subquery
        The query.
    data: pandas.DataFrame
        Data returned from executing the query, as Pandas DataFrame.

    """

    _db: Database
    query: Union[Select, Subquery]
    data: Union[pd.DataFrame, None] = None

    def run(self, limit: Optional[int] = None) -> None:
        """Run the query, and fetch data."""
        if self.data is None:
            self.data = self._db.run_query(self.query, limit=limit)
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
