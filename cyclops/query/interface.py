"""A query interface class to wrap database objects and queries."""

import logging
from typing import List, Literal, Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd
from sqlalchemy.sql.elements import BinaryExpression

import cyclops.query.ops as qo
from cyclops.query.orm import Database
from cyclops.query.util import TableTypes
from cyclops.utils.common import to_list_optional
from cyclops.utils.file import save_dataframe
from cyclops.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


class QueryInterface:
    """An interface dataclass to wrap queries, and run them.

    Parameters
    ----------
    database
        Database object to create ORM, and query data.
    query: cyclops.query.util.TableTypes
        The query.

    """

    def __init__(
        self,
        database: Database,
        query: Union[TableTypes, "QueryInterface"],
    ) -> None:
        """Initialize the QueryInterface object, join and chain operations."""
        self.database = database
        if isinstance(query, QueryInterface):
            self.query = query.query  # type: ignore
        else:
            self.query = query
        self._data = None

    @property
    def data(self) -> Optional[Union[pd.DataFrame, dd.core.DataFrame]]:
        """Get data."""
        return self._data

    def join(
        self,
        join_table: Union[TableTypes, "QueryInterface"],
        on: Optional[
            Union[
                str,
                List[str],
                Tuple[str],
                List[Tuple[str, str]],
            ]
        ] = None,
        on_to_type: Optional[Union[type, List[type]]] = None,
        cond: Optional[BinaryExpression] = None,
        table_cols: Optional[Union[str, List[str]]] = None,
        join_table_cols: Optional[Union[str, List[str]]] = None,
        isouter: Optional[bool] = False,
    ) -> "QueryInterface":
        """Join the query with another table.

        Parameters
        ----------
        join_table
            Table to join with.
        on
            Column(s) to join on.
        on_to_type
            Type(s) to cast the column(s) to join on.
        cond
            Condition to join on.
        table_cols
            Columns to select from the original table.
        join_table_cols
            Columns to select from the joined table.
        isouter
            Whether to perform an outer join.

        Returns
        -------
        QueryInterface
            QueryInterface object with the join operation added.

        """
        on = to_list_optional(on)
        on_to_type = to_list_optional(on_to_type)
        table_cols = to_list_optional(table_cols)
        join_table_cols = to_list_optional(join_table_cols)
        if isinstance(join_table, QueryInterface):
            join_table = join_table.query
        query = qo.Join(
            join_table=join_table,
            on=on,
            on_to_type=on_to_type,
            cond=cond,
            table_cols=table_cols,
            join_table_cols=join_table_cols,
            isouter=isouter,
        )(self.query)

        return QueryInterface(self.database, query)

    def ops(
        self,
        ops: Union[qo.QueryOp, qo.Sequential],
    ) -> "QueryInterface":
        """Chain operations with the query.

        Parameters
        ----------
        ops
            Operations to perform on the query.

        Returns
        -------
        QueryInterface
            QueryInterface object with the operations added.

        """
        query = ops(self.query)

        return QueryInterface(self.database, query)

    def union(
        self,
        other: "QueryInterface",
    ) -> "QueryInterface":
        """Union the query with another query.

        Parameters
        ----------
        other
            The other query to union with.

        Returns
        -------
        QueryInterface
            QueryInterface object with the union operation added.

        """
        query = qo.Union(other.query)(self.query)

        return QueryInterface(self.database, query)

    def union_all(
        self,
        other: "QueryInterface",
    ) -> "QueryInterface":
        """Union all the query with another query.

        Parameters
        ----------
        other
            The other query to union all with.

        Returns
        -------
        QueryInterface
            QueryInterface object with the union all operation added.

        """
        query = qo.Union(other.query, union_all=True)(self.query)

        return QueryInterface(self.database, query)

    def run(
        self,
        limit: Optional[int] = None,
        backend: Literal["pandas", "dask", "datasets"] = "pandas",
        index_col: Optional[str] = None,
        n_partitions: Optional[int] = None,
    ) -> Union[pd.DataFrame, dd.core.DataFrame]:
        """Run the query, and fetch data.

        Parameters
        ----------
        limit
            No. of rows to limit the query return.
        backend
            Backend computing framework to use, pandas or dask or datasets.
        index_col
            Column which becomes the index, and defines the partitioning.
            Should be a indexed column in the SQL server, and any orderable type.
        n_partitions
            Number of partitions. Check dask documentation for additional details.

        Returns
        -------
        pandas.DataFrame or dask.DataFrame or datasets.Dataset
            Query result.

        """
        self._data = self.database.run_query(
            self.query,
            limit=limit,
            backend=backend,
            index_col=index_col,
            n_partitions=n_partitions,
        )

        return self._data

    def save(
        self,
        path: str,
        file_format: Literal["parquet", "csv"] = "parquet",
    ) -> str:
        """Save the query.

        Parameters
        ----------
        path
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
            return save_dataframe(self._data, path, file_format=file_format)

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
