# pylint: disable=too-many-lines

"""Low-level query operations.

This module contains query operation modules such which can be used in high-level query
API functions specific to datasets.

"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import sqlalchemy
from sqlalchemy import and_, cast, extract, func, literal_column, or_, select
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import literal
from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.types import Boolean

# Logging.
from cyclops.query.util import (
    TableTypes,
    apply_to_columns,
    check_timestamp_columns,
    drop_columns,
    ends_with,
    equals,
    filter_columns,
    get_column,
    get_column_names,
    get_columns,
    get_delta_column,
    greater_than,
    has_columns,
    has_substring,
    in_,
    less_than,
    not_equals,
    process_column,
    rename_columns,
    reorder_columns,
    starts_with,
    table_params_to_type,
    trim_columns,
)
from cyclops.utils.common import to_datetime_format, to_list, to_list_optional
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# pylint: disable=too-few-public-methods


@dataclass
class JoinArgs:
    """Arguments for joining tables.

    Parameters
    ----------
    join_table: cyclops.query.util.TableTypes
        Table to join.
    on: list of str or tuple, optional
    on_to_type: list of type, optional
        A list of types to which to convert the on columns before joining. Useful when
        two columns have the same values but in different format, e.g., strings of int.
    cond: BinaryExpression, optional
        Condition on which to join to tables.
    table_cols: str or list of str, optional
        Filters to keep only these columns from the table.
    join_table_cols:
        Filters to keep only these columns from the join_table.
    isouter:
        Flag to say if the join is a left outer join.

    """

    join_table: TableTypes
    on: typing.Optional[  # pylint: disable=invalid-name
        typing.Union[
            str,
            typing.List[str],
            typing.Tuple[str],
            typing.List[typing.Tuple[str, str]],
        ]
    ] = None
    on_to_type: typing.Optional[typing.Union[type, typing.List[type]]] = None
    cond: typing.Optional[BinaryExpression] = None
    table_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None
    join_table_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None
    isouter: typing.Optional[bool] = False

    def __post_init__(self) -> None:
        """Post initialization."""
        self.on = to_list_optional(self.on)
        self.on_to_type = to_list_optional(self.on_to_type)
        self.table_cols = to_list_optional(self.table_cols)
        self.join_table_cols = to_list_optional(self.join_table_cols)


class QueryOp(type):
    """Metaclass type for query operations."""

    def __repr__(cls) -> str:
        """Return the name of the class."""
        return "QueryOp"


def _chain_ops(
    query: Subquery, ops: typing.Union[typing.List[QueryOp], Sequential]
) -> Subquery:
    if isinstance(ops, typing.List):
        for op_ in ops:
            query = op_(query)
    if isinstance(ops, Sequential):
        query = _chain_ops(query, ops.ops)

    return query


class Sequential:
    """Sequential query operations class.

    Chains a list of sequential query operations and executes the final query on a
    table.

    """

    def __init__(self, ops: typing.Union[typing.List[QueryOp], Sequential]):
        """Initialize the Sequential class.

        Parameters
        ----------
        ops: typing.Union[typing.List[QueryOp], Sequential]
            typing.List of query operations to be chained.

        """
        self.ops = ops

    def __call__(self, table: TableTypes) -> Subquery:
        """Execute the query operations on the table.

        Parameters
        ----------
        table: TableTypes
            Table to be queried.

        Returns
        -------
        Subquery
            Query result after chaining the query operations.

        """
        return _chain_ops(table, self.ops)


def _append_if_missing(
    table: TableTypes,
    keep_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    force_include_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
) -> Subquery:
    """Keep only certain columns in a table, but must include certain columns.

    Parameters
    ----------
    table : cyclops.query.util.TableTypes
        Table on which to perform the operation.
    keep_cols: str or list of str, optional
        Columns to keep.
    force_include_cols: str or list of str, optional
        Columns to include (forcefully).

    """
    if keep_cols is None:
        return table
    keep_cols = to_list(keep_cols)
    force_include_cols = to_list(force_include_cols)
    extend_cols = [col for col in force_include_cols if col not in keep_cols]
    keep_cols = extend_cols + keep_cols

    return Keep(keep_cols)(table)


def _none_add(obj1: typing.Any, obj2: typing.Any) -> typing.Any:
    """Add two objects together while ignoring None values.

    If both objects are None, returns None.

    Parameters
    ----------
    obj1: typing.Any
    obj2: typing.Any

    """
    if obj1 is None:
        return obj2
    if obj2 is None:
        return obj1
    return obj1 + obj2


def _process_checks(
    table: TableTypes,
    cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    cols_not_in: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    timestamp_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
) -> Subquery:
    """Perform checks, and possibly alterations, on a table.

    Parameters
    ----------
    table : cyclops.query.util.TableTypes
        Table on which to perform the operation.
    cols: str or list of str, optional
        Columns to check.
    timestamp_cols: str or list of str, optional
        Timestamp columns to check.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Checked and possibly altered table.

    """
    if cols is not None:
        cols = to_list(cols)
        has_columns(table, cols, raise_error=True)

    if cols_not_in is not None:
        cols_not_in = to_list(cols_not_in)
        if has_columns(table, cols_not_in, raise_error=False):
            raise ValueError(f"Cannot specify columns {cols_not_in}.")

    if timestamp_cols is not None:
        timestamp_cols = to_list(timestamp_cols)
        has_columns(table, timestamp_cols, raise_error=True)
        check_timestamp_columns(table, timestamp_cols, raise_error=True)

    return table


@dataclass
class FillNull(metaclass=QueryOp):
    """Fill NULL values with a given value.

    Parameters
    ----------
    cols: str or list of str
        Columns to fill.
    fill_values: typing.Any or list of typing.Any
        Value(s) to fill with.
    new_col_names: str or list of str, optional
        New column name(s) for the filled columns. If not provided,

    """

    cols: typing.Union[str, typing.List[str]]
    fill_values: typing.Union[typing.Any, typing.List[typing.Any]]
    new_col_names: typing.Optional[typing.Union[str, typing.List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Fill NULL values with a given value.

        Parameters
        ----------
        table: TableTypes
            Table on which to perform the operation.

        Returns
        -------
        Subquery
            Table with NULL values filled.

        """
        cols = to_list(self.cols)
        fill_values = to_list(self.fill_values)
        new_col_names = to_list_optional(self.new_col_names)
        if new_col_names:
            if len(cols) != len(new_col_names):
                raise ValueError(
                    """Number of columns to fill and number of new column names
                    must match."""
                )
        table = _process_checks(table, cols=self.cols)
        if len(fill_values) == 1:
            fill_values = fill_values * len(cols)
        for col, fill in zip(cols, fill_values):
            coalesced_col = func.coalesce(table.c[col], fill).label(
                f"coalesced_col_{col}"
            )
            table = select([table, coalesced_col]).subquery()
        if new_col_names:
            for col, new_col in zip(cols, new_col_names):
                table = Rename({f"coalesced_col_{col}": new_col})(table)
        else:
            for col in cols:
                table = drop_columns(table, col)
                table = Rename({f"coalesced_col_{col}": col})(table)

        return table


@dataclass
class Drop(metaclass=QueryOp):
    """Drop some columns.

    Parameters
    ----------
    cols: str or list of str
        Columns to drop.

    """

    cols: typing.Union[str, typing.List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)
        return drop_columns(table, self.cols)


@dataclass
class Rename(metaclass=QueryOp):
    """Rename some columns.

    Parameters
    ----------
    rename_map: dict
        Map from an existing column name to another name.
    check_exists: bool
        Whether to check if all of the keys in the map exist as columns.

    """

    rename_map: typing.Dict[str, str]
    check_exists: bool = True

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if self.check_exists:
            table = _process_checks(table, cols=list(self.rename_map.keys()))
        return rename_columns(table, self.rename_map)


@dataclass
class Substring(metaclass=QueryOp):
    """Get substring of a string column.

    Parameters
    ----------
    col: str
        Name of column which has string, where substring needs
        to be extracted.
    start_index: int
        Start index of substring.
    stop_index: str
        Name of the new column with extracted substring.

    """

    col: str
    start_index: int
    stop_index: int
    new_col_label: str

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col)
        table = select(
            table,
            func.substr(
                get_column(table, self.col), self.start_index, self.stop_index
            ).label(self.new_col_label),
        ).subquery()

        return table


@dataclass
class Reorder(metaclass=QueryOp):
    """Reorder the columns in a table.

    Parameters
    ----------
    cols: list of str
        Complete list of table column names in the new order.

    """

    cols: typing.List[str]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)
        return reorder_columns(table, self.cols)


@dataclass
class ReorderAfter(metaclass=QueryOp):
    """Reorder a number of columns to come after a specified column.

    Parameters
    ----------
    cols: list of str
        Ordered list of column names which will come after a specified column.
    after: str
        Column name for the column after which the other columns will follow.

    """

    cols: typing.Union[str, typing.List[str]]
    after: str

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        self.cols = to_list(self.cols)
        table = _process_checks(table, cols=self.cols + [self.after])
        names = get_column_names(table)
        names = [name for name in names if name not in self.cols]
        name_after_ind = names.index(self.after) + 1
        new_order = names[:name_after_ind] + self.cols + names[name_after_ind:]

        return Reorder(new_order)(table)


@dataclass
class Keep(metaclass=QueryOp):
    """Keep only the specified columns in a table.

    Parameters
    ----------
    cols: str or list of str
        The columns to keep.

    """

    cols: typing.Union[str, typing.List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)
        return filter_columns(table, self.cols)


@dataclass
class Trim(metaclass=QueryOp):
    """Trim the whitespace from some string columns.

    Parameters
    ----------
    cols: str or list of str
        Columns to trim.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    cols: typing.Union[str, typing.List[str]]
    new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)
        return trim_columns(table, self.cols, new_col_labels=self.new_col_labels)


@dataclass
class Literal(metaclass=QueryOp):
    """Add a literal column to a table.

    Parameters
    ----------
    value: any
        Value of the literal, e.g., a string or integer.
    col: str
        Label of the new literal column.

    """

    value: typing.Any
    col: str

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols_not_in=self.col)
        return select(table, literal(self.value).label(self.col)).subquery()


@dataclass
class ExtractTimestampComponent(metaclass=QueryOp):
    """Extract a component such as year or month from a timestamp column.

    Parameters
    ----------
    timestamp_col: str
        Timestamp column from which to extract the time component.
    extract_str: str
        Information to extract, e.g., "year", "month"
    label: str
        Column label for the extracted column.

    """

    timestamp_col: str
    extract_str: str
    label: str

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table, timestamp_cols=self.timestamp_col, cols_not_in=self.label
        )

        table = select(
            table,
            extract(self.extract_str, get_column(table, self.timestamp_col)).label(
                self.label
            ),
        )

        return Cast(self.label, "int")(table)


@dataclass
class AddNumeric(metaclass=QueryOp):
    """Add a numeric value to some columns.

    Parameters
    ----------
    add_to: str or list of str
        Column names specifying to which columns is being added.
    num: int or float
        Adds this value to the add_to columns.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    add_to: typing.Union[str, typing.List[str]]
    num: typing.Union[int, float]
    new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table, cols=self.add_to, cols_not_in=self.new_col_labels
        )
        return apply_to_columns(
            table,
            self.add_to,
            lambda x: x + self.num,
            new_col_labels=self.new_col_labels,
        )


@dataclass
class AddDeltaConstant(metaclass=QueryOp):
    """Construct and add a datetime.timedelta object to some columns.

    Parameters
    ----------
    add_to: str or list of str
        Column names specifying to which columns is being added.
    delta: datetime.timedelta
        A timedelta object.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    add_to: typing.Union[str, typing.List[str]]
    delta: timedelta
    new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
        )

        return apply_to_columns(
            table,
            self.add_to,
            lambda x: x + self.delta,
            new_col_labels=self.new_col_labels,
        )


@dataclass
class AddColumn(metaclass=QueryOp):
    """Add a column to some columns.

    Parameters
    ----------
    add_to: str or list of str
        Column names specifying to which columns is being added.
    col: str
        Column name of column to add to the add_to columns.
    negative: bool, optional
        Subtract the column rather than adding.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Warning
    -------
    Pay attention to column types. Some combinations will work,
    whereas others will not.

    """

    add_to: typing.Union[str, typing.List[str]]
    col: str
    negative: typing.Optional[bool] = False
    new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        # If the column being added is a timestamp column, ensure the others are too
        if check_timestamp_columns(table, self.col):
            table = _process_checks(
                table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
            )
        else:
            table = _process_checks(
                table, cols=self.add_to, cols_not_in=self.new_col_labels
            )

        col = get_column(table, self.col)

        if self.negative:
            return apply_to_columns(
                table,
                self.add_to,
                lambda x: x - col,
                new_col_labels=self.new_col_labels,
            )

        return apply_to_columns(
            table,
            self.add_to,
            lambda x: x + col,
            new_col_labels=self.new_col_labels,
        )


class AddDeltaColumn(metaclass=QueryOp):
    """Construct and add an interval column to some columns.

    Parameters
    ----------
    add_to: str or list of str
        Column names specifying to which columns is being added.
    add: Column
        Column object of type interval.
    negative: bool, optional
        Subtract the object rather than adding.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.
    **delta_kwargs
        The arguments used to create the Interval column.

    """

    def __init__(
        self,
        add_to: typing.Union[str, typing.List[str]],
        negative: typing.Optional[bool] = False,
        new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        **delta_kwargs: typing.Any,
    ) -> None:
        """Initialize."""
        self.add_to = add_to
        self.negative = negative
        self.new_col_labels = new_col_labels
        self.delta_kwargs = delta_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
        )

        delta = get_delta_column(table, **self.delta_kwargs)

        if self.negative:
            return apply_to_columns(
                table,
                self.add_to,
                lambda x: x - delta,
                new_col_labels=self.new_col_labels,
            )

        return apply_to_columns(
            table,
            self.add_to,
            lambda x: x + delta,
            new_col_labels=self.new_col_labels,
        )


@dataclass
class Cast(metaclass=QueryOp):
    """Cast columns to a specified type.

    Currently supporting conversions to str, int, float, date and timestamp.

    Parameters
    ----------
    cols : str or list of str
        Columns to = cast.
    type_ : str
        Name of type to which to convert. Must be supported.

    """

    cols: typing.Union[str, typing.List[str]]
    type_: str

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)

        cast_type_map = {
            "str": "to_str",
            "int": "to_int",
            "float": "to_float",
            "date": "to_date",
            "bool": "to_bool",
            "timestamp": "to_timestamp",
        }

        # Assert that the type inputted is supported
        if self.type_ not in cast_type_map:
            supported_str = ", ".join(list(cast_type_map.keys()))
            raise ValueError(
                f"""Conversion to type {self.type_} not supported. Supporting
                conversion to types {supported_str}"""
            )

        # Cast
        kwargs = {cast_type_map[self.type_]: True}

        return apply_to_columns(
            table,
            self.cols,
            lambda x: process_column(x, **kwargs),  # pylint: disable=unnecessary-lambda
        )


class Union(metaclass=QueryOp):
    """Union two tables.

    Parameters
    ----------
    union_table : cyclops.query.util.TableTypes
        Table to union with the first table.
    union_all : bool, optional
        Whether to use the all keyword in the union.

    """

    def __init__(
        self, union_table: TableTypes, union_all: typing.Optional[bool] = False
    ) -> None:
        """Initialize."""
        self.union_table = union_table
        self.union_all = union_all

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table)
        union_table = _process_checks(self.union_table)

        if self.union_all:
            return select(table).union_all(select(union_table)).subquery()

        return select(table).union(select(union_table)).subquery()


class Join(metaclass=QueryOp):
    """Join a table with another table.

    Parameters
    ----------
    join_table: cyclops.query.util.TableTypes
        Table on which to join.
    on: list of str or tuple, optional
        A list of strings or tuples representing columns on which to join.
        Strings represent columns of same name in both tables. A tuple of
        style (table_col, join_table_col) is used to join on columns of
        different names. Suggested to specify this parameter as opposed to
        cond.
    on_to_type: list of type, optional
        A list of types to which to convert the on columns before joining. Useful when
        two columns have the same values but in different format, e.g., strings of int.
    cond: BinaryExpression, optional
        Condition on which to join to tables.
    table_cols: str or list of str, optional
        Filters to keep only these columns from the table.
    join_table_cols:
        Filters to keep only these columns from the join_table.
    isouter:
        Flag to say if the join is a left outer join.

    Warnings
    --------
    If neither on nor cond parameters are specified, an
    expensive Cartesian product is performed.

    """

    @table_params_to_type(Subquery)
    def __init__(
        self,
        join_table: TableTypes,
        on: typing.Optional[  # pylint: disable=invalid-name
            typing.Union[
                str,
                typing.List[str],
                typing.Tuple[str],
                typing.List[typing.Tuple[str, str]],
            ]
        ] = None,
        on_to_type: typing.Optional[typing.Union[type, typing.List[type]]] = None,
        cond: typing.Optional[BinaryExpression] = None,
        table_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        join_table_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        isouter: typing.Optional[bool] = False,
    ):
        """Initialize."""
        if on is not None and cond is not None:
            raise ValueError("Cannot specify both the 'on' and 'cond' arguments.")

        self.join_table = join_table
        self.cond = cond
        self.on_ = to_list_optional(on)
        self.on_to_type = to_list_optional(on_to_type)
        self.table_cols = to_list_optional(table_cols)
        self.join_table_cols = to_list_optional(join_table_cols)
        self.isouter = isouter

    @table_params_to_type(Subquery)
    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        # Join on the equality of values in columns of same name in both tables
        if self.on_ is not None:
            # Process on columns
            on_table_cols = [
                col_obj if isinstance(col_obj, str) else col_obj[0]
                for col_obj in self.on_
            ]
            on_join_table_cols = [
                col_obj if isinstance(col_obj, str) else col_obj[1]
                for col_obj in self.on_
            ]
            table = _process_checks(
                table, cols=_none_add(self.table_cols, on_table_cols)
            )
            self.join_table = _process_checks(
                self.join_table,
                cols=_none_add(self.join_table_cols, on_join_table_cols),
            )
            # Filter columns, keeping those being joined on
            table = _append_if_missing(table, self.table_cols, on_table_cols)
            self.join_table = _append_if_missing(
                self.join_table, self.join_table_cols, on_join_table_cols
            )
            # Perform type conversions if given
            if self.on_to_type is not None:
                for i, type_ in enumerate(self.on_to_type):
                    table = Cast(on_table_cols[i], type_)(table)
                    self.join_table = Cast(on_join_table_cols[i], type_)(
                        self.join_table
                    )
            cond = and_(
                *[
                    get_column(table, on_table_cols[i])
                    == get_column(self.join_table, on_join_table_cols[i])
                    for i in range(len(on_table_cols))
                ]
            )
            table = select(table.join(self.join_table, cond, isouter=self.isouter))

        else:
            # Filter columns
            if self.table_cols is not None:
                table = Keep(self.table_cols)(table)
            if self.join_table_cols is not None:
                self.join_table = Keep(self.table_cols)(self.join_table)  # type: ignore

            # Join on a specified condition
            if self.cond is not None:
                table = select(
                    table.join(  # type: ignore
                        self.join_table, self.cond, isouter=self.isouter
                    )
                )

            # Join on no condition, i.e., a Cartesian product
            else:
                LOGGER.warning("A Cartesian product has been queried.")
                table = select(table, self.join_table)

        # Filter to include no duplicate columns
        return select(
            *[col for col in table.subquery().columns if "%(" not in col.name]
        ).subquery()


class ConditionEquals(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on being equal, or not equal, to some value.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    value: any
        Value to equal.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        value: typing.Any,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):
        """Initialize."""
        self.col = col
        self.value = value
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = equals(
            get_column(table, self.col), self.value, True, True, **self.cond_kwargs
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionGreaterThan(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on greater than (or equal), to some value.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    value: any
        Value greater than.
    equal: bool, default=False
        Include equality to the value.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        value: typing.Any,
        equal: bool = False,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):
        """Initialize."""
        self.col = col
        self.value = value
        self.equal = equal
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = greater_than(
            get_column(table, self.col),
            self.value,
            True,
            True,
            self.equal,
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionLessThan(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on less than (or equal), to some value.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    value: any
        Value greater than.
    equal: bool, default=False
        Include equality to the value.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        value: typing.Any,
        equal: bool = False,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):
        """Initialize."""
        self.col = col
        self.value = value
        self.equal = equal
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = less_than(
            get_column(table, self.col),
            self.value,
            True,
            True,
            self.equal,
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionRegexMatch(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on matching a regular expression.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    regex: str
        Regular expression to match.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.

    """

    def __init__(
        self,
        col: str,
        regex: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        """Initialize."""
        self.col = col
        self.regex = regex
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = get_column(table, self.col).regexp_match(self.regex)
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionIn(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on having a value in list of values.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    values: any or list of any
        Values in which the column value must be.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        values: typing.Union[typing.Any, typing.List[typing.Any]],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):
        """Initialize."""
        self.col = col
        self.values = values
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = in_(
            get_column(table, self.col),
            to_list(self.values),
            True,
            True,
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionSubstring(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows on based on having substrings.

    Can be specified whether it must have any or all of the specified substrings.
    This makes no difference when only one substring is provided

    Parameters
    ----------
    col: str
        Column name on which to condition.
    substrings: any
        Substrings.
    any_: bool, default=True
        If true, the row must have just one of the substrings. If false, it must
        have all of the substrings.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        substrings: typing.Union[str, typing.List[str]],
        any_: bool = True,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):  # pylint: disable=too-many-arguments
        """Initialize."""
        self.col = col
        self.substrings = to_list(substrings)
        self.any_ = any_
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        conds = [
            has_substring(get_column(table, self.col), sub, True, **self.cond_kwargs)
            for sub in self.substrings
        ]

        if self.any_:
            cond = or_(*conds)
        else:
            cond = and_(*conds)

        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionStartsWith(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on starting with some string.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    string: any
        String.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        string: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):
        """Initialize."""
        self.col = col
        self.string = string
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = starts_with(
            get_column(table, self.col), self.string, True, True, **self.cond_kwargs
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionEndsWith(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on ending with some string.

    Parameters
    ----------
    col: str
        Column name on which to condition.
    string: any
        String.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        typing.Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        string: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ):
        """Initialize."""
        self.col = col
        self.string = string
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = ends_with(
            get_column(table, self.col), self.string, True, True, **self.cond_kwargs
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionInYears(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp column being in a list of years.

    Parameters
    ----------
    timestamp_col: str
        Timestamp column name.
    years: int or list of int
        Years in which the timestamps must be.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.

    """

    def __init__(
        self,
        timestamp_col: str,
        years: typing.Union[int, typing.List[int]],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.years = years
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table, cols=self.timestamp_col, cols_not_in=self.binarize_col
        )
        cond = in_(
            extract("year", get_column(table, self.timestamp_col)),
            to_list(self.years),
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionInMonths(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp being in a list of years.

    Parameters
    ----------
    timestamp_col: str
        Timestamp column name.
    months: int or list of int
        Months in which the timestamps must be.
    not_: bool, default=False
        Take negation of condition.
    binarize_col: str, optional
        If specified, create a Boolean column of name binarize_col instead of filtering.

    """

    def __init__(
        self,
        timestamp_col: str,
        months: typing.Union[int, typing.List[int]],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.months = months
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table, cols=self.timestamp_col, cols_not_in=self.binarize_col
        )
        cond = in_(
            extract("month", get_column(table, self.timestamp_col)),
            to_list(self.months),
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionBeforeDate(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp being before some date.

    Parameters
    ----------
    timestamp_col: str
        Timestamp column name.
    timestamp: str or datetime.datetime
        A datetime object or str in YYYY-MM-DD format.

    """

    def __init__(self, timestamp_col: str, timestamp: typing.Union[str, datetime]):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.timestamp = timestamp

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, timestamp_cols=self.timestamp_col)

        if isinstance(self.timestamp, str):
            timestamp = to_datetime_format(self.timestamp)
        else:
            timestamp = self.timestamp

        return (
            select(table)
            .where(get_column(table, self.timestamp_col) <= timestamp)
            .subquery()
        )


class ConditionAfterDate(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp being after some date.

    Parameters
    ----------
    timestamp_col: str
        Timestamp column name.
    timestamp: str or datetime.datetime
        A datetime object or str in YYYY-MM-DD format.

    """

    def __init__(self, timestamp_col: str, timestamp: typing.Union[str, datetime]):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.timestamp = timestamp

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, timestamp_cols=self.timestamp_col)

        if isinstance(self.timestamp, str):
            timestamp = to_datetime_format(self.timestamp)
        else:
            timestamp = self.timestamp

        return (
            select(table)
            .where(get_column(table, self.timestamp_col) >= timestamp)
            .subquery()
        )


@dataclass
class Limit(metaclass=QueryOp):  # pylint: disable=too-few-public-methods
    """Limit the number of rows returned in a query.

    Parameters
    ----------
    number: int
        Number of rows to return in the limit.

    """

    number: int

    @table_params_to_type(Select)
    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        return table.limit(self.number).subquery()  # type: ignore


@dataclass
class RandomizeOrder(metaclass=QueryOp):
    """Randomize order of table rows.

    Useful when the data is ordered, so certain rows cannot
    be seen or analyzed when limited.

    Warnings
    --------
    Becomes quite slow on large tables.

    """

    @table_params_to_type(Subquery)
    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        return select(table).order_by(func.random()).subquery()


@dataclass
class DropNulls(metaclass=QueryOp):
    """Remove rows with null values in some specified columns.

    Parameters
    ----------
    cols: str or list of str
        Columns in which, if a value is null, the corresponding row
        is removed.

    """

    cols: typing.Union[str, typing.List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        self.cols = to_list(self.cols)
        table = _process_checks(table, cols=self.cols)

        cond = and_(*[not_equals(get_column(table, col), None) for col in self.cols])
        return select(table).where(cond).subquery()


@dataclass
class Apply(metaclass=QueryOp):
    """Apply a function to column(s).

    The function must take a sqlalchemy column object and also return a column object.

    Parameters
    ----------
    cols: str or list of str
        Column(s) to apply the function to.
    func: typing.Callable
        Function that takes in single sqlalchemy column object and returns a column
        after applying the function.
    new_cols: str or list of str, optional
        New column name(s) after function is applied to the specified column(s).

    """

    cols: typing.Union[str, typing.List[str]]
    func: typing.Callable[[sqlalchemy.sql.schema.Column], sqlalchemy.sql.schema.Column]
    new_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        arg_count = self.func.__code__.co_argcount
        self.new_cols = to_list(self.new_cols)
        if arg_count != 1:
            if len(self.new_cols) != 1:
                raise ValueError(
                    """Only one result column possible, and needed when
                computing function using multiple column args."""
                )
            cols = get_columns(table, self.cols)
            result_col = self.func(*cols).label(self.new_cols[0])
            return select(table).add_columns(result_col).subquery()

        return apply_to_columns(table, self.cols, self.func, self.new_cols)


@dataclass
class OrderBy(metaclass=QueryOp):
    """Order, or sort, the rows of a table by some columns.

    Parameters
    ----------
    cols: str or list of str
        Columns by which to order.
    ascending: bool or list of bool
        Whether to order each columns by ascending (True) or descending (False).
        If not provided, orders all by ascending.

    """

    cols: typing.Union[str, typing.List[str]]
    ascending: typing.Optional[typing.Union[bool, typing.List[bool]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        self.cols = to_list(self.cols)
        ascending = to_list_optional(self.ascending)
        table = _process_checks(table, cols=self.cols)

        if ascending is None:
            ascending = [True] * len(self.cols)
        else:
            if len(ascending) != len(self.cols):
                raise ValueError(
                    "If ascending is specified. Must specify for all columns."
                )

        order_cols = [
            col if ascending[i] else col.desc()
            for i, col in enumerate(get_columns(table, self.cols))
        ]

        return select(table).order_by(*order_cols).subquery()


@dataclass
class GroupByAggregate(metaclass=QueryOp):
    """Aggregate over a group by object.

    Parameters
    ----------
    groupby_cols: str or list of str
        Columns by which to group.
    aggfuncs: dict
        Specify a dictionary of key-value pairs:
        column name: aggfunc string or
        column name: (aggfunc string, new column label)
        This labelling prevents the aggregation of the same column using multiple
        aggregation functions.
    aggseps: dict, optional
        Specify a dictionary of key-value pairs:
        column name: string_aggfunc separator
        If string_agg used as aggfunc for a column, then a separator must be provided
        for the same column.

    Examples
    --------
    >>> GroupByAggregate("person_id", {"person_id": "count"})(table)
    >>> GroupByAggregate("person_id", {"person_id": ("count", "visit_count")})(table)
    >>> GroupByAggregate("person_id", {"lab_name": "string_agg"}, {"lab_name": ", "})(table)  # noqa: E501, pylint: disable=line-too-long
    >>> GroupByAggregate("person_id", {"lab_name": ("string_agg", "lab_name_agg"}, {"lab_name": ", "})(table)  # noqa: E501, pylint: disable=line-too-long

    """

    groupby_cols: typing.Union[str, typing.List[str]]
    aggfuncs: typing.Union[
        typing.Dict[str, typing.Sequence[str]], typing.Dict[str, str]
    ]
    aggseps: typing.Dict[str, str] = field(default_factory=dict)

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        str_to_aggfunc = {
            "sum": func.sum,
            "average": func.avg,
            "min": func.min,
            "max": func.max,
            "count": func.count,
            "median": func.percentile_disc(0.5).within_group,
            "string_agg": func.string_agg,
        }

        aggfunc_tuples = list(self.aggfuncs.items())
        aggfunc_cols = [item[0] for item in aggfunc_tuples]
        aggfunc_strs = [
            item[1] if isinstance(item[1], str) else item[1][0]
            for item in aggfunc_tuples
        ]

        # If not specified, aggregate column names default to that of
        # the column being aggregated over
        aggfunc_names = [
            aggfunc_cols[i] if isinstance(item[1], str) else item[1][1]
            for i, item in enumerate(aggfunc_tuples)
        ]

        groupby_names = to_list(self.groupby_cols)
        table = _process_checks(table, cols=groupby_names + aggfunc_cols)

        # Error checking
        for i, aggfunc_str in enumerate(aggfunc_strs):
            if aggfunc_str not in str_to_aggfunc:
                allowed_strs = ", ".join(list(str_to_aggfunc.keys()))
                raise ValueError(
                    f"Invalid aggfuncs specified. Allowed values are {allowed_strs}."
                )
            if aggfunc_str == "string_agg":
                if not bool(self.aggseps) or aggfunc_cols[i] not in self.aggseps:
                    raise ValueError(
                        f"""Column {aggfunc_cols[i]} needs to be aggregated as string, must specify a separator!"""  # noqa: E501, pylint: disable=line-too-long
                    )

        all_names = groupby_names + aggfunc_names
        if len(all_names) != len(set(all_names)):
            raise ValueError(
                """Duplicate column names were found. Try naming aggregated columns
                to avoid this issue."""
            )

        # Perform group by
        groupby_cols = get_columns(table, groupby_names)
        to_agg_cols = get_columns(table, aggfunc_cols)
        agg_cols = []
        for i, to_agg_col in enumerate(to_agg_cols):
            if aggfunc_strs[i] == "string_agg":
                agg_col = str_to_aggfunc[aggfunc_strs[i]](
                    to_agg_col, literal_column(f"'{self.aggseps[aggfunc_cols[i]]}'")
                )
            else:
                agg_col = str_to_aggfunc[aggfunc_strs[i]](to_agg_col)
            agg_cols.append(agg_col.label(aggfunc_names[i]))

        return select(*groupby_cols, *agg_cols).group_by(*groupby_cols).subquery()


@dataclass
class Distinct(metaclass=QueryOp):
    """Get distinct rows.

    Parameters
    ----------
    cols: str or list of str
        Columns to use for distinct.

    Examples
    --------
    >>> Distinct("person_id")(table)
    >>> Distinct(["person_id", "visit_id"])(table)

    """

    cols: typing.Union[str, typing.List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        cols = to_list(self.cols)
        table = _process_checks(table, cols=cols)
        return select(table).distinct(*get_columns(table, cols)).subquery()
