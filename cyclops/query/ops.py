"""Query operations."""

from __future__ import annotations

import logging
import operator
import typing
from abc import abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from itertools import islice

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


# ruff: noqa: W505


def _addindent(s_: str, num_spaces: int = 4) -> str:
    """Add spaces to a string except the first line.

    Parameters
    ----------
    s_
        String to add spaces to.
    num_spaces
        Number of spaces to add.

    Returns
    -------
    str
        String with spaces added.

    """
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)  # type: ignore

    return first + "\n" + s  # type: ignore


class QueryOp:
    """Base class for query operations."""

    _ops: typing.Dict[str, "QueryOp"]

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__setattr__("_ops", OrderedDict())

    @abstractmethod
    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> Subquery:
        """Implement a calling function."""
        pass

    def _add_op(self, name: str, op_: "QueryOp") -> None:
        """Add a child operation to the current query operation.

        The query op can be accessed as an attribute using the given name.

        Parameters
        ----------
        name
            Name of the child op. The child op can be accessed from this op using
            the given name
        op_
            Child op to be added to the parent query op.

        """
        if not isinstance(op_, QueryOp) and op_ is not None:
            raise TypeError("{} is not a QueryOp subclass".format(str(op_)))
        if not isinstance(name, str):
            raise TypeError("Query op name should be a string")
        if hasattr(self, name) and name not in self._ops:
            raise KeyError("Attribute '{}' already exists".format(name))
        if "." in name:
            raise KeyError('Query op name can\'t contain ".", got: {}'.format(name))
        if name == "":
            raise KeyError('Query op name can\'t be empty string ""')
        self._ops[name] = op_

    def _get_ops(self) -> typing.Iterator["QueryOp"]:
        """Return an iterator over the child operations.

        Returns
        -------
        typing.Iterator[QueryOp]
            Iterator over the child operations.

        """
        for _, op_ in self._ops.items():
            yield op_

    def _get_name(self) -> str:
        """Get the name of the query op.

        Returns
        -------
        str
            Name of the query op.

        """
        return self.__class__.__name__

    def __setattr__(self, name: str, value: "QueryOp") -> None:
        """Set an attribute.

        Parameters
        ----------
        name
            Name of the attribute.
        value
            Value of the attribute.

        """
        ops = self.__dict__.get("_ops")
        if isinstance(value, QueryOp):
            if ops is None:
                raise AttributeError("Can't assign op before QueryOp.__init__() call")
            ops[name] = value
        elif ops is not None and name in ops:
            if value is not None:
                raise TypeError(
                    "Cannot assign '{}' as child op '{}' " "(QueryOp or None expected)",
                )
            ops[name] = value
        else:
            super().__setattr__(name, value)

    def _extra_repr(self) -> str:
        """Set the extra representation of the query op.

        To print customized extra information, you should re-implement
        this method in your own query ops. Both single-line and multi-line
        strings are acceptable.

        Returns
        -------
        str
            Extra representation of the query op.

        """
        return ""

    def __repr__(self) -> str:
        """Return the string representation of the query op.

        Returns
        -------
        str
            String representation of the query op.

        """
        extra_lines = []
        extra_repr = self._extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, op_ in self._ops.items():
            mod_str = repr(op_)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"

        return main_str

    def __getattr__(self, name: str) -> "QueryOp":
        """Get an attribute.

        Parameters
        ----------
        name
            Name of the attribute.

        Returns
        -------
        QueryOp
            The child operation with the given name.

        """
        if name in self._ops:
            return self._ops[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'",
        )


def _chain_ops(
    query: Subquery,
    ops: typing.Iterator[QueryOp],
) -> Subquery:
    """Chain query ops.

    Parameters
    ----------
    query
        Query to chain the ops to.
    ops
        Query ops to chain.

    Returns
    -------
    Subquery
        Query with the ops chained.

    """
    for op_ in ops:
        if isinstance(op_, Sequential):
            query = _chain_ops(query, op_._get_ops())
        elif isinstance(op_, QueryOp):
            query = op_(query)

    return query


class Sequential(QueryOp):
    """Sequential query operations class.

    Chains a sequence of query operations and executes the final query on a table.

    Examples
    --------
    >>> Sequential(Drop(["col1", "col2"]), ...)
    >>> Sequential([Drop(["col1", "col2"]), ...])

    """

    @typing.overload
    def __init__(self, *ops: QueryOp) -> None:
        ...

    @typing.overload
    def __init__(self, ops: typing.List[QueryOp]) -> None:
        ...

    @typing.overload
    def __init__(self, op: OrderedDict[str, QueryOp]) -> None:
        ...

    def __init__(self, *args: QueryOp) -> None:  # type: ignore
        """Initialize the class.

        Parameters
        ----------
        args
            Query operations to be chained sequentially.

        """
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, op_ in args[0].items():
                self._add_op(key, op_)
        elif len(args) == 1 and isinstance(args[0], list):
            for idx, op_ in enumerate(args[0]):
                self._add_op(str(idx), op_)
        else:
            for idx, op_ in enumerate(args):
                self._add_op(str(idx), op_)

    def __len__(self) -> int:
        """Return the number of query ops in the Sequential.

        Returns
        -------
        int
            Number of query ops in the Sequential.

        """
        return len(self._ops)

    def __iter__(self) -> typing.Iterator[QueryOp]:
        """Return an iterator over the query ops.

        Returns
        -------
        typing.Iterator[QueryOp]
            Iterator over the query ops.

        """
        return iter(self._ops.values())

    def __add__(self, other: "Sequential") -> "Sequential":
        """Add two Sequential objects.

        Parameters
        ----------
        other
            Sequential object to be added.

        Returns
        -------
        Sequential
            Sequential object with the two Sequential objects chained.

        """
        if isinstance(other, Sequential):
            ret = Sequential()
            for op_ in self:
                ret.append(op_)
            for op_ in other:
                ret.append(op_)
            return ret
        raise ValueError(
            "Add operator supports only objects "
            "of Sequential class, but {} is given.".format(str(type(other))),
        )

    def __iadd__(self, other: "Sequential") -> "Sequential":
        """Add two Sequential objects inplace.

        Parameters
        ----------
        other
            Sequential object to be added.

        Returns
        -------
        Sequential
            Sequential object with the two Sequential objects chained.

        """
        if isinstance(other, Sequential):
            offset = len(self)
            for i, op_ in enumerate(other):
                self._add_op(str(i + offset), op_)
            return self
        raise ValueError(
            "Add operator supports only objects "
            "of Sequential class, but {} is given.".format(str(type(other))),
        )

    def _get_item_by_idx(
        self,
        iterator: typing.Iterator[typing.Any],
        idx: int,
    ) -> typing.Any:
        """Get the idx-th item of the iterator.

        Parameters
        ----------
        iterator
            Iterator to get the item from.
        idx
            Index of the item to get.

        Returns
        -------
        QueryOp
            The idx-th item of the iterator.

        """
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size

        return next(islice(iterator, idx, None))

    def __getitem__(
        self,
        idx: typing.Union[slice, int],
    ) -> typing.Any:
        """Get the idx-th item of the sequential query op.

        Parameters
        ----------
        idx
            Index of the item to get.

        Returns
        -------
        Sequential or QueryOp
            The idx-th item of the sequential query op.

        """
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._ops.items())[idx]))

        return self._get_item_by_idx(self._ops.values(), idx)  # type: ignore

    def __setitem__(self, idx: int, op_: QueryOp) -> None:
        """Set the idx-th item of the sequential query op.

        Parameters
        ----------
        idx
            Index of the item to set.
        op_
            Query op to set.

        """
        key: str = self._get_item_by_idx(self._ops.keys(), idx)  # type: ignore
        return setattr(self, key, op_)

    def __delitem__(self, idx: typing.Union[slice, int]) -> None:
        """Delete the idx-th item of the sequential query op.

        Parameters
        ----------
        idx
            Index of the item to delete.

        """
        if isinstance(idx, slice):
            for key in list(self._ops.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._ops.keys(), idx)  # type: ignore
            delattr(self, key)
        str_indices = [str(i) for i in range(len(self._ops))]
        self._ops = OrderedDict(list(zip(str_indices, self._ops.values())))

    def append(self, op_: QueryOp) -> "Sequential":
        """Append a given query op to the end.

        Parameters
        ----------
        op_
            Query op to append.

        Returns
        -------
        Sequential
            Sequential object with the query op appended.

        """
        self._add_op(str(len(self)), op_)
        return self

    def pop(self, key: typing.Union[int, slice]) -> QueryOp:
        """Pop the query op at the given index.

        Parameters
        ----------
        key
            Index of the query op to pop.

        Returns
        -------
        QueryOp
            Popped query op.

        """
        v = self[key]
        del self[key]

        return v  # type: ignore

    def insert(self, index: int, op_: QueryOp) -> "Sequential":
        """Insert a given query op at the given index.

        Parameters
        ----------
        index
            Index to insert the query op at.
        op_
            Query op to insert.

        Returns
        -------
        Sequential
            Sequential object with the query op inserted.

        """
        if not isinstance(op_, QueryOp):
            raise AssertionError("Module should be of type: {}".format(QueryOp))
        n = len(self._ops)
        if not (-n <= index <= n):
            raise IndexError("Index out of range: {}".format(index))
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._ops[str(i)] = self._ops[str(i - 1)]
        self._ops[str(index)] = op_

        return self

    def extend(self, sequential: "Sequential") -> "Sequential":
        """Extend the sequential query op with another sequential query op.

        Parameters
        ----------
        sequential
            Sequential object to extend with.

        Returns
        -------
        Sequential
            Sequential object with the other sequential query op extended.

        """
        for op_ in sequential:
            self.append(op_)

        return self

    @table_params_to_type(Subquery)
    def __call__(self, table: TableTypes) -> Subquery:
        """Execute the query operations on the table.

        Parameters
        ----------
        table
            Table to be queried.

        Returns
        -------
        Subquery
            Query result after chaining the query operations.

        """
        return _chain_ops(table, self._get_ops())


def _append_if_missing(
    table: TableTypes,
    keep_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    force_include_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
) -> Subquery:
    """Keep only certain columns in a table, but must include certain columns.

    Parameters
    ----------
    table
        Table on which to perform the operation.
    keep_cols
        Columns to keep.
    force_include_cols
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
    obj1
        First object to add.
    obj2
        Second object to add.

    Returns
    -------
    typing.Any
        Result of adding the two objects.

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
    table
        Table on which to perform the operation.
    cols
        Columns to check.
    timestamp_cols
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


class FillNull(QueryOp):
    """Fill NULL values with a given value.

    Parameters
    ----------
    cols
        Columns to fill.
    fill_values
        Value(s) to fill with.
    new_col_names
        New column name(s) for the filled columns. If not provided,

    Examples
    --------
    >>> FillNull("col1", 0)(table)
    >>> FillNull(["col1", "col2"], [0, 1])(table)
    >>> FillNull(["col1", "col2"], [0, 1], ["col1_new", "col2_new"])(table)

    """

    def __init__(
        self,
        cols: typing.Union[str, typing.List[str]],
        fill_values: typing.Union[typing.Any, typing.List[typing.Any]],
        new_col_names: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    ) -> None:
        super().__init__()
        self.cols = cols
        self.fill_values = fill_values
        self.new_col_names = new_col_names

    def __call__(self, table: TableTypes) -> Subquery:
        """Fill NULL values with a given value.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        Subquery
            Table with NULL values filled.

        """
        cols = to_list(self.cols)
        fill_values = to_list(self.fill_values)
        new_col_names = to_list_optional(self.new_col_names)
        if new_col_names and len(cols) != len(new_col_names):
            raise ValueError(
                """Number of columns to fill and number of new column names
                    must match.""",
            )
        table = _process_checks(table, cols=self.cols)
        if len(fill_values) == 1:
            fill_values = fill_values * len(cols)
        for col, fill in zip(cols, fill_values):
            coalesced_col = func.coalesce(table.c[col], fill).label(
                f"coalesced_col_{col}",
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


class Drop(QueryOp):
    """Drop some columns.

    Parameters
    ----------
    cols
        Columns to drop.

    Examples
    --------
    >>> Drop("col1")(table)
    >>> Drop(["col1", "col2"])(table)

    """

    def __init__(self, cols: typing.Union[str, typing.List[str]]) -> None:
        super().__init__()
        self.cols = cols

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)

        return drop_columns(table, self.cols)


class Rename(QueryOp):
    """Rename some columns.

    Parameters
    ----------
    rename_map
        Map from an existing column name to another name.
    check_exists
        Whether to check if all of the keys in the map exist as columns.

    Examples
    --------
    >>> Rename({"col1": "col1_new"})(table)

    """

    def __init__(self, rename_map: typing.Dict[str, str], check_exists: bool = True):
        super().__init__()
        self.rename_map = rename_map
        self.check_exists = check_exists

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if self.check_exists:
            table = _process_checks(table, cols=list(self.rename_map.keys()))

        return rename_columns(table, self.rename_map)


class Substring(QueryOp):
    """Get substring of a string column.

    Parameters
    ----------
    col
        Name of column which has string, where substring needs
        to be extracted.
    start_index
        Start index of substring.
    stop_index
        Stop index of substring.
    new_col_name
        Name of the new column with extracted substring.

    Examples
    --------
    >>> Substring("col1", 0, 2, "col1_substring")(table)

    """

    def __init__(
        self,
        col: str,
        start_index: int,
        stop_index: int,
        new_col_label: typing.Optional[str] = None,
    ):
        super().__init__()
        self.col = col
        self.start_index = start_index
        self.stop_index = stop_index
        self.new_col_label = new_col_label

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.col, cols_not_in=self.new_col_label)

        return apply_to_columns(
            table,
            self.col,
            lambda x: func.substr(
                process_column(x, to_str=True),
                self.start_index,
                self.stop_index,
            ),
            new_col_labels=self.new_col_label,
        )


class Reorder(QueryOp):
    """Reorder the columns in a table.

    Parameters
    ----------
    cols
        Complete list of table column names in the new order.

    Examples
    --------
    >>> Reorder(["col2", "col1"])(table)

    """

    def __init__(self, cols: typing.List[str]):
        super().__init__()
        self.cols = cols

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)

        return reorder_columns(table, self.cols)


class ReorderAfter(QueryOp):
    """Reorder a number of columns to come after a specified column.

    Parameters
    ----------
    cols
        Ordered list of column names which will come after a specified column.
    after
        Column name for the column after which the other columns will follow.

    Examples
    --------
    >>> ReorderAfter(["col2", "col1"], "col3")(table)

    """

    def __init__(self, cols: typing.Union[str, typing.List[str]], after: str):
        super().__init__()
        self.cols = cols
        self.after = after

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
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


class Keep(QueryOp):
    """Keep only the specified columns in a table.

    Parameters
    ----------
    cols
        The columns to keep.

    Examples
    --------
    >>> Keep("col1")(table)
    >>> Keep(["col1", "col2"])(table)

    """

    def __init__(self, cols: typing.Union[str, typing.List[str]]):
        super().__init__()
        self.cols = cols

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)

        return filter_columns(table, self.cols)


class Trim(QueryOp):
    """Trim the whitespace from some string columns.

    Parameters
    ----------
    cols
        Columns to trim.
    new_col_labels
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Examples
    --------
    >>> Trim("col1")(table)
    >>> Trim(["col1", "col2"])(table)
    >>> Trim("col1", "col1_trimmed")(table)
    >>> Trim(["col1", "col2"], ["col1_trimmed", "col2_trimmed"])(table)

    """

    def __init__(
        self,
        cols: typing.Union[str, typing.List[str]],
        new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    ):
        super().__init__()
        self.cols = cols
        self.new_col_labels = new_col_labels

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols=self.cols)

        return trim_columns(table, self.cols, new_col_labels=self.new_col_labels)


class Literal(QueryOp):
    """Add a literal column to a table.

    Parameters
    ----------
    value
        Value of the literal, e.g., a string or integer.
    col
        Label of the new literal column.

    Examples
    --------
    >>> Literal(1, "col1")(table)

    """

    def __init__(self, value: typing.Any, col: str):
        super().__init__()
        self.value = value
        self.col = col

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(table, cols_not_in=self.col)

        return select(table, literal(self.value).label(self.col)).subquery()


class ExtractTimestampComponent(QueryOp):
    """Extract a component such as year or month from a timestamp column.

    Parameters
    ----------
    timestamp_col
        Timestamp column from which to extract the time component.
    extract_str
        Information to extract, e.g., "year", "month"
    label
        Column label for the extracted column.

    Examples
    --------
    >>> ExtractTimestampComponent("col1", "year", "year")(table)
    >>> ExtractTimestampComponent("col1", "month", "month")(table)

    """

    def __init__(self, timestamp_col: str, extract_str: str, label: str):
        super().__init__()
        self.timestamp_col = timestamp_col
        self.extract_str = extract_str
        self.label = label

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table,
            timestamp_cols=self.timestamp_col,
            cols_not_in=self.label,
        )
        table = select(
            table,
            extract(self.extract_str, get_column(table, self.timestamp_col)).label(
                self.label,
            ),
        )

        return Cast(self.label, "int")(table)


class AddNumeric(QueryOp):
    """Add a numeric value to some columns.

    Parameters
    ----------
    add_to
        Column names specifying to which columns is being added.
    add
        Adds this value to the add_to columns.
    new_col_labels
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Examples
    --------
    >>> AddNumeric("col1", 1)(table)
    >>> AddNumeric(["col1", "col2"], 1)(table)
    >>> AddNumeric("col1", 1, "col1_plus_1")(table)
    >>> AddNumeric(["col1", "col2"], 1, ["col1_plus_1", "col2_plus_1"])(table)
    >>> AddNumeric(["col1", "col2"], [1, 2.2])(table)

    """

    def __init__(
        self,
        add_to: typing.Union[str, typing.List[str]],
        add: typing.Union[int, float, typing.List[int], typing.List[float]],
        new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    ):
        super().__init__()
        self.add_to = add_to
        self.add = add
        self.new_col_labels = new_col_labels

    def _gen_lambda(
        self,
        add: typing.Union[int, float],
    ) -> typing.Callable[[sqlalchemy.sql.schema.Column], sqlalchemy.sql.schema.Column]:
        """Generate the lambda function."""
        return lambda x: x + add

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table,
            cols=self.add_to,
            cols_not_in=self.new_col_labels,
        )
        self.add_to = to_list(self.add_to)
        if isinstance(self.add, (int, float)) and len(self.add_to) > 1:
            add = [self.add] * len(self.add_to)
        elif isinstance(self.add, (int, float)) and len(self.add_to) == 1:
            add = [self.add]
        elif isinstance(self.add, list) and len(self.add_to) != len(self.add):
            raise ValueError(
                "Length of add_to and add must be the same if add is a list.",
            )

        return apply_to_columns(
            table,
            self.add_to,
            [self._gen_lambda(add_num) for add_num in add],
            new_col_labels=self.new_col_labels,
        )


class AddDeltaConstant(QueryOp):
    """Construct and add a datetime.timedelta object to some columns.

    Parameters
    ----------
    add_to
        Column names specifying to which columns is being added.
    delta
        A timedelta object.
    new_col_labels
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Examples
    --------
    >>> AddDeltaConstant("col1", datetime.timedelta(days=1))(table)
    >>> AddDeltaConstant(["col1", "col2"], datetime.timedelta(days=1))(table)
    >>> AddDeltaConstant("col1", datetime.timedelta(days=1), "col1_plus_1")(table)

    """

    def __init__(
        self,
        add_to: typing.Union[str, typing.List[str]],
        delta: timedelta,
        new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    ):
        super().__init__()
        self.add_to = add_to
        self.delta = delta
        self.new_col_labels = new_col_labels

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table,
            timestamp_cols=self.add_to,
            cols_not_in=self.new_col_labels,
        )

        return apply_to_columns(
            table,
            self.add_to,
            lambda x: x + self.delta,
            new_col_labels=self.new_col_labels,
        )


class AddColumn(QueryOp):
    """Add a column to some columns.

    Parameters
    ----------
    add_to
        Column names specifying to which columns is being added.
    col
        Column name of column to add to the add_to columns.
    negative
        Subtract the column rather than adding.
    new_col_labels
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Examples
    --------
    >>> AddColumn("col1", "col2")(table)
    >>> AddColumn(["col1", "col2"], "col3")(table)
    >>> AddColumn("col1", "col2", negative=True)(table)
    >>> AddColumn("col1", "col2", "col1_plus_col2")(table)
    >>> AddColumn(["col1", "col2"], "col3", ["col1_plus_col3", "col2_plus_col3"])(table)

    Warning
    -------
    Pay attention to column types. Some combinations will work,
    whereas others will not.

    """

    def __init__(
        self,
        add_to: typing.Union[str, typing.List[str]],
        col: str,
        negative: typing.Optional[bool] = False,
        new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    ):
        super().__init__()
        self.add_to = add_to
        self.col = col
        self.negative = negative
        self.new_col_labels = new_col_labels

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        # If the column being added is a timestamp column, ensure the others are too
        if check_timestamp_columns(table, self.col):
            table = _process_checks(
                table,
                timestamp_cols=self.add_to,
                cols_not_in=self.new_col_labels,
            )
        else:
            table = _process_checks(
                table,
                cols=self.add_to,
                cols_not_in=self.new_col_labels,
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


class AddDeltaColumn(QueryOp):
    """Construct and add an interval column to some columns.

    Parameters
    ----------
    add_to
        Column names specifying to which columns is being added.
    negative
        Subtract the object rather than adding.
    new_col_labels
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.
    **delta_kwargs
        The arguments used to create the Interval column.

    Examples
    --------
    >>> AddDeltaColumn("col1", "col2")(table)
    >>> AddDeltaColumn(["col1", "col2"], "col3")(table)
    >>> AddDeltaColumn("col1", "col2", negative=True)(table)
    >>> AddDeltaColumn("col1", "col2", "col1_plus_col2")(table)

    """

    def __init__(
        self,
        add_to: typing.Union[str, typing.List[str]],
        negative: typing.Optional[bool] = False,
        new_col_labels: typing.Optional[typing.Union[str, typing.List[str]]] = None,
        **delta_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.add_to = add_to
        self.negative = negative
        self.new_col_labels = new_col_labels
        self.delta_kwargs = delta_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = _process_checks(
            table,
            timestamp_cols=self.add_to,
            cols_not_in=self.new_col_labels,
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


class Cast(QueryOp):
    """Cast columns to a specified type.

    Currently supporting conversions to str, int, float, date, bool and timestamp.

    Parameters
    ----------
    cols
        Columns to cast.
    type_
        Name of type to which to convert. Must be supported.

    Examples
    --------
    >>> Cast("col1", "str")(table)
    >>> Cast(["col1", "col2"], "int")(table)
    >>> Cast("col1", "float")(table)
    >>> Cast("col1", "date")(table)
    >>> Cast("col1", "bool")(table)
    >>> Cast("col1", "timestamp")(table)

    """

    def __init__(self, cols: typing.Union[str, typing.List[str]], type_: str):
        super().__init__()
        self.cols = cols
        self.type_ = type_

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
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
                conversion to types {supported_str}""",
            )
        # Cast
        kwargs = {cast_type_map[self.type_]: True}

        return apply_to_columns(
            table,
            self.cols,
            lambda x: process_column(x, **kwargs),
        )


class Union(QueryOp):
    """Union two tables.

    Parameters
    ----------
    union_table
        Table to union with the first table.
    union_all
        Whether to use the all keyword in the union.

    Examples
    --------
    >>> Union(table2)(table1)
    >>> Union(table2, union_all=True)(table1)

    """

    def __init__(
        self,
        union_table: TableTypes,
        union_all: typing.Optional[bool] = False,
    ):
        super().__init__()
        self.union_table = union_table
        self.union_all = union_all

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
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


class Join(QueryOp):
    """Join a table with another table.

    Parameters
    ----------
    join_table
        Table on which to join.
    on
        A list of strings or tuples representing columns on which to join.
        Strings represent columns of same name in both tables. A tuple of
        style (table_col, join_table_col) is used to join on columns of
        different names. Suggested to specify this parameter as opposed to
        cond.
    on_to_type
        A list of types to which to convert the on columns before joining. Useful when
        two columns have the same values but in different format, e.g., strings of int.
    cond
        Condition on which to join to tables.
    table_cols
        Filters to keep only these columns from the table.
    join_table_cols
        Filters to keep only these columns from the join_table.
    isouter
        Flag to say if the join is a left outer join.

    Examples
    --------
    >>> Join(table2, on=["col1", ("col2", "col3")], on_to_type=[int, str])(table1)
    >>> Join(table2, table_cols=["col1", "col2"])(table1)
    >>> Join(table2, join_table_cols=["col1", "col2"])(table1)

    Warnings
    --------
    If neither on nor cond parameters are specified, an
    expensive Cartesian product is performed.

    """

    @table_params_to_type(Subquery)
    def __init__(
        self,
        join_table: TableTypes,
        on: typing.Optional[
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
    ) -> None:
        super().__init__()
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
        table
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
                table,
                cols=_none_add(self.table_cols, on_table_cols),
            )
            self.join_table = _process_checks(
                self.join_table,
                cols=_none_add(self.join_table_cols, on_join_table_cols),
            )
            # Filter columns, keeping those being joined on
            table = _append_if_missing(table, self.table_cols, on_table_cols)
            self.join_table = _append_if_missing(
                self.join_table,
                self.join_table_cols,
                on_join_table_cols,
            )
            # Perform type conversions if given
            if self.on_to_type is not None:
                for i, type_ in enumerate(self.on_to_type):
                    table = Cast(on_table_cols[i], type_)(table)
                    self.join_table = Cast(on_join_table_cols[i], type_)(
                        self.join_table,
                    )
            cond = and_(
                *[
                    get_column(table, on_table_cols[i])
                    == get_column(self.join_table, on_join_table_cols[i])
                    for i in range(len(on_table_cols))
                ],
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
                        self.join_table,
                        self.cond,
                        isouter=self.isouter,
                    ),
                )
            # Join on no condition, i.e., a Cartesian product
            else:
                LOGGER.warning("A Cartesian product has been queried.")
                table = select(table, self.join_table)

        # Filter to include no duplicate columns
        return select(
            *[col for col in table.subquery().columns if "%(" not in col.name],
        ).subquery()


class ConditionEquals(QueryOp):
    """Filter rows based on being equal, or not equal, to some value.

    Parameters
    ----------
    col
        Column name on which to condition.
    value
        Value to equal.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionEquals("col1", 1)(table)
    >>> ConditionEquals("col1", 1, binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        value: typing.Any,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.value = value
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = equals(
            get_column(table, self.col),
            self.value,
            True,
            True,
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionGreaterThan(QueryOp):
    """Filter rows based on greater than (or equal), to some value.

    Parameters
    ----------
    col
        Column name on which to condition.
    value
        Value greater than.
    equal
        Include equality to the value.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionGreaterThan("col1", 1)(table)
    >>> ConditionGreaterThan("col1", 1, binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        value: typing.Any,
        equal: bool = False,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.value = value
        self.equal = equal
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
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
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionLessThan(QueryOp):
    """Filter rows based on less than (or equal), to some value.

    Parameters
    ----------
    col
        Column name on which to condition.
    value
        Value greater than.
    equal
        Include equality to the value.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionLessThan("col1", 1)(table)
    >>> ConditionLessThan("col1", 1, binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        value: typing.Any,
        equal: bool = False,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.value = value
        self.equal = equal
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
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
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionRegexMatch(QueryOp):
    """Filter rows based on matching a regular expression.

    Parameters
    ----------
    col
        Column name on which to condition.
    regex
        Regular expression to match.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.

    Examples
    --------
    >>> ConditionRegexMatch("col1", ".*")(table)
    >>> ConditionRegexMatch("col1", ".*", binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        regex: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        super().__init__()
        self.col = col
        self.regex = regex
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = get_column(table, self.col).regexp_match(self.regex)
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionIn(QueryOp):
    """Filter rows based on having a value in list of values.

    Parameters
    ----------
    col
        Column name on which to condition.
    values
        Values in which the column value must be.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionIn("col1", [1, 2])(table)
    >>> ConditionIn("col1", [1, 2], binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        values: typing.Union[typing.Any, typing.List[typing.Any]],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.values = values
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
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
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionSubstring(QueryOp):
    """Filter rows on based on having substrings.

    Can be specified whether it must have any or all of the specified substrings.
    This makes no difference when only one substring is provided

    Parameters
    ----------
    col
        Column name on which to condition.
    substrings
        Substrings.
    any_
        If true, the row must have just one of the substrings. If false, it must
        have all of the substrings.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionSubstring("col1", ["a", "b"])(table)
    >>> ConditionSubstring("col1", ["a", "b"], any_=False)(table)
    >>> ConditionSubstring("col1", ["a", "b"], binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        substrings: typing.Union[str, typing.List[str]],
        any_: bool = True,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.substrings = to_list(substrings)
        self.any_ = any_
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        conds = [
            has_substring(get_column(table, self.col), sub, True, **self.cond_kwargs)
            for sub in self.substrings
        ]
        cond = or_(*conds) if self.any_ else and_(*conds)
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionStartsWith(QueryOp):
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
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionStartsWith("col1", "a")(table)
    >>> ConditionStartsWith("col1", "a", binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        string: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.string = string
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = starts_with(
            get_column(table, self.col),
            self.string,
            True,
            True,
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionEndsWith(QueryOp):
    """Filter rows based on ending with some string.

    Parameters
    ----------
    col
        Column name on which to condition.
    string
        String to end with.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.
    **cond_kwargs
        Optional keyword arguments for processing the condition.

    Examples
    --------
    >>> ConditionEndsWith("col1", "a")(table)
    >>> ConditionEndsWith("col1", "a", binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        col: str,
        string: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
        **cond_kwargs: typing.Any,
    ) -> None:
        super().__init__()
        self.col = col
        self.string = string
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = ends_with(
            get_column(table, self.col),
            self.string,
            True,
            True,
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionInYears(QueryOp):
    """Filter rows based on a timestamp column being in a list of years.

    Parameters
    ----------
    timestamp_col
        Timestamp column name.
    years
        Years in which the timestamps must be.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.

    Examples
    --------
    >>> ConditionInYears("col1", [2019, 2020])(table)
    >>> ConditionInYears("col1", 2019)(table)
    >>> ConditionInYears("col1", 2019, binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        timestamp_col: str,
        years: typing.Union[int, typing.List[int]],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        super().__init__()
        self.timestamp_col = timestamp_col
        self.years = years
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(
            table,
            cols=self.timestamp_col,
            cols_not_in=self.binarize_col,
        )
        cond = in_(
            extract("year", get_column(table, self.timestamp_col)),
            to_list(self.years),
        )
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionInMonths(QueryOp):
    """Filter rows based on a timestamp being in a list of years.

    Parameters
    ----------
    timestamp_col
        Timestamp column name.
    months
        Months in which the timestamps must be.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.

    Examples
    --------
    >>> ConditionInMonths("col1", [1, 2])(table)
    >>> ConditionInMonths("col1", 1)(table)
    >>> ConditionInMonths("col1", 1, binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        timestamp_col: str,
        months: typing.Union[int, typing.List[int]],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        super().__init__()
        self.timestamp_col = timestamp_col
        self.months = months
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(
            table,
            cols=self.timestamp_col,
            cols_not_in=self.binarize_col,
        )
        cond = in_(
            extract("month", get_column(table, self.timestamp_col)),
            to_list(self.months),
        )
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionBeforeDate(QueryOp):
    """Filter rows based on a timestamp being before some date.

    Parameters
    ----------
    timestamp_col
        Timestamp column name.
    timestamp
        A datetime object or str in YYYY-MM-DD format.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.

    Examples
    --------
    >>> ConditionBeforeDate("col1", "2020-01-01")(table)
    >>> ConditionBeforeDate("col1", datetime.datetime(2020, 1, 1))(table)
    >>> ConditionBeforeDate("col1", "2020-01-01", binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        timestamp_col: str,
        timestamp: typing.Union[str, datetime],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        super().__init__()
        self.timestamp_col = timestamp_col
        self.timestamp = timestamp
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, timestamp_cols=self.timestamp_col)
        if isinstance(self.timestamp, str):
            timestamp = to_datetime_format(self.timestamp)
        else:
            timestamp = self.timestamp
        cond = get_column(table, self.timestamp_col) <= timestamp
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionAfterDate(QueryOp):
    """Filter rows based on a timestamp being after some date.

    Parameters
    ----------
    timestamp_col
        Timestamp column name.
    timestamp
        A datetime object or str in YYYY-MM-DD format.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.

    Examples
    --------
    >>> ConditionAfterDate("col1", "2020-01-01")(table)
    >>> ConditionAfterDate("col1", datetime.datetime(2020, 1, 1))(table)
    >>> ConditionAfterDate("col1", "2020-01-01", binarize_col="col1_bool")(table)

    """

    def __init__(
        self,
        timestamp_col: str,
        timestamp: typing.Union[str, datetime],
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        super().__init__()
        self.timestamp_col = timestamp_col
        self.timestamp = timestamp
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, timestamp_cols=self.timestamp_col)
        if isinstance(self.timestamp, str):
            timestamp = to_datetime_format(self.timestamp)
        else:
            timestamp = self.timestamp
        cond = get_column(table, self.timestamp_col) >= timestamp
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionLike(QueryOp):
    """Filter rows by a LIKE condition.

    Parameters
    ----------
    col
        Column to filter on.
    pattern
        Pattern to filter on.
    not_
        Take negation of condition.
    binarize_col
        If specified, create a Boolean column of name binarize_col instead of filtering.

    Examples
    --------
    >>> ConditionLike("lab_name", "HbA1c")(table)
    >>> ConditionLike("lab_name", "HbA1c", not_=True)(table)
    >>> ConditionLike("lab_name", "HbA1c", binarize_col="lab_name_bool")(table)

    """

    def __init__(
        self,
        col: str,
        pattern: str,
        not_: bool = False,
        binarize_col: typing.Optional[str] = None,
    ):
        super().__init__()
        self.col = col
        self.pattern = pattern
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if return_cond and self.binarize_col:
            raise ValueError(
                "Cannot return condition and binarize column simultaneously.",
            )
        table = _process_checks(table, cols=self.col)
        cond = get_column(table, self.col).like(self.pattern)
        if self.not_:
            cond = cond._negate()
        if return_cond:
            return cond
        if self.binarize_col is not None:
            return select(
                table,
                cast(cond, Boolean).label(self.binarize_col),
            ).subquery()

        return select(table).where(cond).subquery()


class Or(QueryOp):
    """Combine multiple condition query ops using an OR.

    Parameters
    ----------
    cond_ops
        Condition Query ops to combine.

    Examples
    --------
    >>> Or(ConditionLike("lab_name", "HbA1c"), ConditionIn("name", ["John", "Jane"]))
    >>> Or([ConditionLike("lab_name", "HbA1c"), ConditionIn("name", ["John", "Jane"])])

    """

    def __init__(self, *cond_ops: typing.Union[QueryOp, typing.List[QueryOp]]):
        super().__init__()
        self.cond_ops = cond_ops

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        ops = []
        for cond_op in self.cond_ops:
            if isinstance(cond_op, list):
                if len(self.cond_ops) != 1:
                    raise ValueError("Cannot combine multiple lists of conditions.")
                ops = [op(table, return_cond=True) for op in cond_op]
            if isinstance(cond_op, QueryOp):
                if len(self.cond_ops) == 1:
                    return cond_op(table, return_cond=return_cond)
                ops.append(cond_op(table, return_cond=True))
        cond = or_(*ops)
        if return_cond:
            return cond

        return select(table).where(cond).subquery()


class And(QueryOp):
    """Combine multiple condition query ops using an And.

    Parameters
    ----------
    ops
        Query ops to combine.

    Examples
    --------
    >>> And([ConditionLike("lab_name", "HbA1c"), ConditionIn("name", ["John", "Jane"])])
    >>> And(ConditionLike("lab_name", "HbA1c"), ConditionIn("name", ["John", "Jane"]))

    """

    def __init__(self, *cond_ops: typing.Union[QueryOp, typing.List[QueryOp]]):
        super().__init__()
        self.cond_ops = cond_ops

    def __call__(self, table: TableTypes, return_cond: bool = False) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.
        return_cond
            Return the condition instead of filtering.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        ops = []
        for cond_op in self.cond_ops:
            if isinstance(cond_op, list):
                if len(self.cond_ops) != 1:
                    raise ValueError("Cannot combine multiple lists of conditions.")
                ops = [op(table, return_cond=True) for op in cond_op]
            if isinstance(cond_op, QueryOp):
                if len(self.cond_ops) == 1:
                    return cond_op(table, return_cond=return_cond)
                ops.append(cond_op(table, return_cond=True))
        cond = and_(*ops)
        if return_cond:
            return cond

        return select(table).where(cond).subquery()


class Limit(QueryOp):
    """Limit the number of rows returned in a query.

    Parameters
    ----------
    number
        Number of rows to return in the limit.

    Examples
    --------
    >>> Limit(10)(table)

    """

    def __init__(self, number: int):
        super().__init__()
        self.number = number

    @table_params_to_type(Select)
    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        return table.limit(self.number).subquery()  # type: ignore


class RandomizeOrder(QueryOp):
    """Randomize order of table rows.

    Useful when the data is ordered, so certain rows cannot
    be seen or analyzed when limited.

    Examples
    --------
    >>> RandomizeOrder()(table)

    Warnings
    --------
    Becomes quite slow on large tables.

    """

    @table_params_to_type(Subquery)
    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        return select(table).order_by(func.random()).subquery()


class DropNulls(QueryOp):
    """Remove rows with null values in some specified columns.

    Parameters
    ----------
    cols
        Columns in which, if a value is null, the corresponding row is removed.

    Examples
    --------
    >>> DropNulls("col1")(table)
    >>> DropNulls(["col1", "col2"])(table)

    """

    def __init__(self, cols: typing.Union[str, typing.List[str]]):
        super().__init__()
        self.cols = cols

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
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


class Apply(QueryOp):
    """Apply function(s) to column(s).

    The function can take a sqlalchemy column object and also return a column object.
    It can also take multiple columns and return a single column or multiple columns.
    If multiple functions are provided, it is assumed that each function is applied to
    each input column.

    Parameters
    ----------
    cols
        Column(s) to apply the function to.
    funcs
        Function(s) that takes in sqlalchemy column(s) object and returns column(s)
        after applying the function or list of functions to apply to each column.
    new_cols
        New column name(s) after function is applied to the specified column(s).

    Examples
    --------
    >>> Apply("col1", lambda x: x + 1)(table)
    >>> Apply(["col1", "col2"], [lambda x: x + 1, lambda x: x + 2])(table)
    >>> Apply("col1", lambda x: x + 1, new_cols="col1_new")(table)
    >>> Apply(["col1", "col2"], lambda x, y: x + y, new_cols="col1_new")(table)
    >>> Apply(["col1", "col2"], lambda x, y: (x + y, x - y), new_cols=["col1_new", "col2_new"])(table)  # noqa: E501, pylint: disable=line-too-long

    """

    def __init__(
        self,
        cols: typing.Union[str, typing.List[str]],
        funcs: typing.Union[
            typing.Callable[
                [sqlalchemy.sql.schema.Column],
                sqlalchemy.sql.schema.Column,
            ],
            typing.List[
                typing.Callable[
                    [sqlalchemy.sql.schema.Column],
                    sqlalchemy.sql.schema.Column,
                ]
            ],
        ],
        new_cols: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    ):
        super().__init__()
        self.cols = cols
        self.funcs = funcs
        self.new_cols = new_cols

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        self.new_cols = to_list(self.new_cols)
        if isinstance(self.funcs, list):
            if len(self.funcs) != len(self.cols):
                raise ValueError(
                    "Number of functions must be equal to number of columns.",
                )
            if len(self.new_cols) != len(self.cols):
                raise ValueError(
                    "Number of new columns must be equal to number of columns.",
                )
        if callable(self.funcs):
            cols = get_columns(table, self.cols)
            result_cols = [
                self.funcs(*cols).label(new_col) for new_col in self.new_cols
            ]  # noqa: E501

            return select(table).add_columns(*result_cols).subquery()

        return apply_to_columns(table, self.cols, self.funcs, self.new_cols)


class OrderBy(QueryOp):
    """Order, or sort, the rows of a table by some columns.

    Parameters
    ----------
    cols
        Columns by which to order.
    ascending
        Whether to order each columns by ascending (True) or descending (False).
        If not provided, orders all by ascending.

    Examples
    --------
    >>> OrderBy("col1")(table)
    >>> OrderBy(["col1", "col2"])(table)
    >>> OrderBy(["col1", "col2"], [True, False])(table)
    >>> OrderBy(["col1", "col2"], True)(table)

    """

    def __init__(
        self,
        cols: typing.Union[str, typing.List[str]],
        ascending: typing.Optional[typing.Union[bool, typing.List[bool]]] = None,
    ):
        super().__init__()
        self.cols = cols
        self.ascending = ascending

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
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
        elif len(ascending) != len(self.cols):
            raise ValueError(
                "If ascending is specified. Must specify for all columns.",
            )
        order_cols = [
            col if ascending[i] else col.desc()
            for i, col in enumerate(get_columns(table, self.cols))
        ]

        return select(table).order_by(*order_cols).subquery()


class GroupByAggregate(QueryOp):
    """Aggregate over a group by object.

    Parameters
    ----------
    groupby_cols
        Columns by which to group.
    aggfuncs
        Specify a dictionary of key-value pairs:
        column name: aggfunc string or
        column name: (aggfunc string, new column label)
        This labelling prevents the aggregation of the same column using multiple
        aggregation functions.
    aggseps
        Specify a dictionary of key-value pairs:
        column name: string_aggfunc separator
        If string_agg used as aggfunc for a column, then a separator must be provided
        for the same column.

    Examples
    --------
    >>> GroupByAggregate("person_id", {"person_id": "count"})(table)
    >>> GroupByAggregate("person_id", {"person_id": ("count", "visit_count")})(table)
    >>> GroupByAggregate("person_id", {"lab_name": "string_agg"}, {"lab_name": ", "})(table)
    >>> GroupByAggregate("person_id", {"lab_name": ("string_agg", "lab_name_agg"}, {"lab_name": ", "})(table)

    """

    def __init__(
        self,
        groupby_cols: typing.Union[str, typing.List[str]],
        aggfuncs: typing.Union[
            typing.Dict[str, typing.Sequence[str]],
            typing.Dict[str, str],
        ],
        aggseps: typing.Optional[typing.Dict[str, str]] = None,
    ):
        super().__init__()
        self.groupby_cols = groupby_cols
        self.aggfuncs = aggfuncs
        if aggseps is None:
            aggseps = {}
        self.aggseps = aggseps

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
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
            "median": func.percentile_cont(0.5).within_group,
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
                    f"Invalid aggfuncs specified. Allowed values are {allowed_strs}.",
                )
            if aggfunc_str == "string_agg" and (
                not bool(self.aggseps) or aggfunc_cols[i] not in self.aggseps
            ):
                raise ValueError(
                    f"""Column {aggfunc_cols[i]} needs to be aggregated as string, must specify a separator!""",  # noqa: E501
                )

        all_names = groupby_names + aggfunc_names
        if len(all_names) != len(set(all_names)):
            raise ValueError(
                """Duplicate column names were found. Try naming aggregated columns
                to avoid this issue.""",
            )

        # Perform group by
        groupby_cols = get_columns(table, groupby_names)
        to_agg_cols = get_columns(table, aggfunc_cols)
        agg_cols = []
        for i, to_agg_col in enumerate(to_agg_cols):
            if aggfunc_strs[i] == "string_agg":
                agg_col = str_to_aggfunc[aggfunc_strs[i]](
                    to_agg_col,
                    literal_column(f"'{self.aggseps[aggfunc_cols[i]]}'"),
                )
            else:
                agg_col = str_to_aggfunc[aggfunc_strs[i]](to_agg_col)
            agg_cols.append(agg_col.label(aggfunc_names[i]))

        return select(*groupby_cols, *agg_cols).group_by(*groupby_cols).subquery()


class Distinct(QueryOp):
    """Get distinct rows.

    Parameters
    ----------
    cols
        Columns to use for distinct.

    Examples
    --------
    >>> Distinct("person_id")(table)
    >>> Distinct(["person_id", "visit_id"])(table)

    """

    def __init__(self, cols: typing.Union[str, typing.List[str]]):
        super().__init__()
        self.cols = cols

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Parameters
        ----------
        table
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        cols = to_list(self.cols)
        table = _process_checks(table, cols=cols)

        return select(table).distinct(*get_columns(table, cols)).subquery()
