"""A query interface class to wrap database objects and queries."""

from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
from sqlalchemy.sql.selectable import Select, Subquery

from cyclops.orm import Database


@dataclass
class QueryInterface:
    """An interface dataclass to actually wrap queries, and run them.

    Attributes
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

    def __repr__(self) -> str:
        """Print SQL when printing the QueryInterface object."""
        return str(self.query)
