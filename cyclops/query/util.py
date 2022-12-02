"""Utility functions for querying."""

# mypy: ignore-errors
# pylint: disable=too-many-lines

import logging
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, List, Optional, Union

import sqlalchemy
from sqlalchemy import Float, Integer, Interval, String, cast, func, select
from sqlalchemy.dialects.postgresql.base import DATE, TIMESTAMP
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import ColumnClause
from sqlalchemy.sql.schema import Column, Table
from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.types import Boolean

from cyclops.utils.common import to_list, to_list_optional
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)

COLUMN_OBJECTS = [Column, ColumnClause]


@dataclass
class DBSchema:
    """Database schema wrapper.

    Parameters
    ----------
    name: str
        Name of schema.
    data: sqlalchemy.sql.schema.MetaData
        Metadata for schema.

    """

    name: str
    data: sqlalchemy.sql.schema.MetaData


@dataclass
class DBTable:
    """Database table wrapper.

    Parameters
    ----------
    name: str
        Name of table.
    data: sqlalchemy.sql.schema.Table
        Metadata for schema.

    """

    name: str
    data: sqlalchemy.sql.schema.MetaData


TABLE_OBJECTS = [Table, Select, Subquery, DBTable]
TableTypes = Union[Select, Subquery, Table, DBTable]


def _to_subquery(table: TableTypes) -> Subquery:
    """Convert a table from a table type object to the Subquery type.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        Table to convert.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The converted table.

    """
    if isinstance(table, Subquery):
        return table

    if isinstance(table, Select):
        return table.subquery()

    if isinstance(table, Table):
        return select(table).subquery()

    if isinstance(table, DBTable):
        return select(table.data).subquery()

    raise ValueError(
        f"""Table has type {type(table)}, but must have one of the
        following types: {", ".join(TABLE_OBJECTS)}"""
    )


def _to_select(table: TableTypes) -> Select:
    """Convert a table from a table type object to the Select type.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        Table to convert.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The converted table.

    """
    if isinstance(table, Select):
        return table

    if isinstance(table, Subquery):
        return select(table)

    if isinstance(table, Table):
        return select(table)

    if isinstance(table, DBTable):
        return select(table.data)

    raise ValueError(
        f"""Table has type {type(table)}, but must have one of the
        following types: {", ".join(TABLE_OBJECTS)}"""
    )


def param_types_to_type(relevant_types: List[Any], to_type_fn: Callable) -> Callable:
    """Convert TableTypes parameters to a specified type.

    A decorator which processes a function's arguments by taking all
    parameters with type in relevant_types and converting them using
    some to_type_fn function. Non-relevant types are left alone.

    Parameters
    ----------
    relevant_types : list
        Types to process.
    to_type_fn : Callable
        Function to process the relevant types

    Returns
    -------
    Callable
        The processed function.

    """

    def decorator(func_: Callable) -> Callable:
        """Decorate function to convert TableTypes parameters to a specified type."""

        @wraps(func_)
        def wrapper_func(*args, **kwargs) -> Callable:
            # Convert relevant arguments.
            args = list(args)
            for i, arg in enumerate(args):
                if type(arg) in relevant_types:
                    args[i] = to_type_fn(arg)

            # Convert relevant keyword arguments.
            kwargs = dict(kwargs)
            for key, kwarg in kwargs.items():
                if type(kwarg) in relevant_types:
                    kwargs[key] = to_type_fn(kwarg)

            return func_(*tuple(args), **kwargs)

        return wrapper_func

    return decorator


def table_params_to_type(to_type: TableTypes) -> Callable:
    """Decorate to convert TableTypes params to a specified type.

    Parameters
    ----------
    to_type: cyclops.query.util.TableTypes
        The type to which to convert.

    Returns
    -------
    Callable
        The processed function.

    """
    # Dictionary mapping query type -> query type conversion function.
    table_to_type_fn_map = {
        Subquery: _to_subquery,
        Select: _to_select,
        Table: lambda x: x,
        DBTable: lambda x: x,
    }
    if to_type not in TABLE_OBJECTS:
        raise ValueError(f"to_type must be in {TABLE_OBJECTS}")

    to_type_fn = table_to_type_fn_map[to_type]

    return param_types_to_type(TABLE_OBJECTS, to_type_fn)


@table_params_to_type(Subquery)
def get_column(
    table: TableTypes,
    col: str,
):
    """Extract a column object from a table by name.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table with the column.
    col: str
        Name of column to extract.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        The corresponding column in the table.

    """
    col_names = get_column_names(table)
    if col not in col_names:
        raise ValueError(f"Table does not contain column {col}")

    return table.c[col_names.index(col)]


@table_params_to_type(Subquery)
def filter_columns(
    table: TableTypes,
    cols: Union[str, List[str]],
) -> Subquery:
    """Filter a table, keeping only the specified columns.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table with the column.
    cols: str or list of str
        Name of columns to keep.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Table with only the specified columns.

    """
    cols = to_list(cols)
    col_names = get_column_names(table)
    filtered = []
    for col in cols:
        if col not in col_names:
            continue
        filtered.append(table.c[col_names.index(col)])

    return select(filtered).subquery()


@table_params_to_type(Subquery)
def get_columns(
    table: TableTypes,
    cols: Union[str, List[str]],
):
    """Extract a number of columns from the table.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    cols: str or list of str
        Names of columns to extract.

    Returns
    -------
    list of sqlalchemy.sql.schema.Column
        The corresponding columns in the table.

    """
    return [get_column(table, col) for col in to_list(cols)]


@table_params_to_type(Subquery)
def get_column_names(table: TableTypes):
    """Extract column names from a table.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.

    Returns
    -------
    list of str
        The table column names.

    """
    return [c.name for c in table.columns]


@table_params_to_type(Subquery)
def has_columns(
    table: TableTypes, cols: Union[str, List[str]], raise_error: bool = False
):
    """Check whether a table has all of the specified columns.

    Parameters
    ----------
    table : cyclops.query.util.TableTypes
        Table to check.
    cols: str or list of str
        Required columns.
    raise_error: bool
        Whether to raise an error if the required columns are not found.

    Returns
    -------
    bool
        True if all required columns are present, otherwise False.

    """
    cols = to_list(cols)
    required_set = set(cols)
    columns = set(get_column_names(table))
    present = required_set.issubset(columns)

    if raise_error and not present:
        missing = required_set - columns
        raise ValueError(f"Missing required columns {', '.join(missing)}.")

    return present


@table_params_to_type(Subquery)
def assert_table_has_columns(*args, **kwargs) -> Callable:
    """Assert that TableTypes params have the necessary columns.

    assert_table_has_columns(["A", "B"], None) is equivalent to
    assert_table_has_columns(["A", "B"]) but may be necessary when
    wanting to check, assert_table_has_columns(["A"], None, ["C"])

    Can also check keyword arguments, e.g., optional queries,
    assert_table_has_columns(["A"], kwarg_table=["D"])

    Parameters
    ----------
    *args
        Ordered arguments corresponding to the function's table-type args.
    **kwargs
        Keyword arguments corresponding to the function's table-type kwargs.

    Returns
    -------
    Callable
        Decorator function.

    """

    def decorator(func_: Callable) -> Callable:
        @wraps(func_)
        def wrapper_func(*fn_args, **fn_kwargs) -> Callable:
            # Check only the table arguments
            table_args = [i for i in fn_args if isinstance(i, Subquery)]

            assert len(args) <= len(table_args)

            for i, arg in enumerate(args):
                if arg is None:  # Can specify None to skip over checking a query
                    continue
                has_columns(table_args[i], arg, raise_error=True)

            for key, required_cols in kwargs.items():
                # If an optional table is not provided, or is None,
                # it is skipped
                if key not in fn_kwargs:
                    continue

                if fn_kwargs[key] is None:
                    continue

                assert isinstance(fn_kwargs[key], Subquery)
                has_columns(fn_kwargs[key], required_cols, raise_error=True)

            return func_(*fn_args, **fn_kwargs)

        return wrapper_func

    return decorator


@table_params_to_type(Subquery)
def drop_columns(
    table: TableTypes,
    drop_cols: Union[str, List[str]],
) -> Subquery:
    """Drop, or remove, some columns from a table.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    col : str or list of str
        Names of columns to drop.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The corresponding table with columns dropped.

    """
    drop_cols = get_columns(table, drop_cols)

    return select(*[c for c in table.c if c not in drop_cols]).subquery()


@table_params_to_type(Subquery)
def rename_columns(table: TableTypes, rename_map: dict) -> Subquery:
    """Rename a table's columns.

    Rename the table's columns according to a dictionary of strings,
    where the key is the current name, and the value is the replacement.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    d : dict
        Dictionary mapping current column names (key) to new ones (value).

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The corresponding table with columns renamed.

    """
    return select(
        *[
            c.label(rename_map[c.name]) if c.name in rename_map else c
            for c in table.columns
        ]
    ).subquery()


@table_params_to_type(Subquery)
def reorder_columns(table: TableTypes, cols: List[str]) -> Subquery:
    """Reorder a table's columns.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table to reorder.
    cols : list of str
        New order of columns, which must include all existing columns.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The reordered table.

    """
    # Get the old/new column names.
    old_order = get_column_names(table)
    new_order = [c.name for c in get_columns(table, cols)]

    # Make sure we have exactly the same set of old/new column names.
    if not set(old_order) == set(new_order):
        old_order_print = ", ".join(old_order)
        new_order_print = ", ".join(new_order)
        raise ValueError(
            f"""Must specify all columns {old_order_print}
            to re-order, not {new_order_print}."""
        )

    # Reorder the columns.
    new_cols = []
    for col in new_order:
        new_cols.append(table.c[old_order.index(col)])

    return select(*new_cols).subquery()


@table_params_to_type(Subquery)
def apply_to_columns(
    table: TableTypes,
    col_names: Union[str, List[str]],
    func_: Callable,
    new_col_labels: Optional[Union[str, List[str]]] = None,
) -> Subquery:
    """Apply a function to some columns.

    This function can change existing columns or create new
    columns depending on whether new_col_labels is specified.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    col_names: str or list of str
        Columns to which to apply the function.
    func_: Callable
        Function to apply to the columns, where it takes an column
        as its only parameter and returns another column object.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The table with function applied.

    """
    col_names = to_list(col_names)
    new_col_labels = to_list_optional(new_col_labels)
    cols = get_columns(table, col_names)

    if new_col_labels is None:
        # Apply to existing columns
        prev_order = get_column_names(table)
        table = select(table).add_columns(
            *[
                func_(col).label("__" + col_names[i] + "__")
                for i, col in enumerate(cols)
            ]
        )
        rename = {"__" + name + "__": name for name in col_names}
        table = drop_columns(table, col_names)
        table = rename_columns(table, rename)
        table = reorder_columns(table, prev_order)
    else:
        # Apply to new columns
        new_cols = [func_(col).label(new_col_labels[i]) for i, col in enumerate(cols)]
        table = select(table).add_columns(*new_cols)

    return _to_subquery(table)


def trim_columns(
    table: TableTypes,
    cols: Union[str, List[str]],
    new_col_labels: Optional[Union[str, List[str]]] = None,
) -> Subquery:
    """Trim, or strip, specified columns.

    Trimming refers to the removal of leading/trailing whitespace.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    cols: str or list of str
        Names of columns to trim.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The table with the specified columns trimmed.

    """
    return apply_to_columns(
        table,
        cols,
        lambda x: process_column(x, to_str=True, trim=True),
        new_col_labels=new_col_labels,
    )


def process_elem(elem: Any, **kwargs: bool) -> Any:
    """Preprocess some basic object such as an integer, float, or string.

    Parameters
    ----------
    elem: any
        An element such as an integer, float, or string.
    **kwargs : dict, optional
        Preprocessing keyword arguments.

    Returns
    -------
    Any
        The preprocessed element.

    """
    # Extract kwargs.
    lower = kwargs.get("lower", False)
    trim = kwargs.get("trim", False)
    to_str = kwargs.get("to_str", False)
    to_int = kwargs.get("to_int", False)
    to_float = kwargs.get("to_float", False)
    to_bool = kwargs.get("to_bool", False)

    # Convert to string.
    if to_str:
        elem = str(elem)

    # If a string.
    if isinstance(elem, str):
        if lower:
            elem = elem.lower()

        if trim:
            elem = elem.strip()

    if to_int:
        elem = int(elem)

    if to_float:
        elem = float(elem)

    if to_bool:
        elem = bool(elem)

    return elem


def process_list(lst: Union[Any, List[Any]], **kwargs: bool) -> List[Any]:
    """Preprocess a list of elements.

    Parameters
    ----------
    lst : any or list of any
        A list of elements such as integers, floats, or strings.
    **kwargs : dict, optional
        Preprocessing keyword arguments.

    Returns
    -------
    Any
        The preprocessed element.

    """
    # Convert potentially non-list variable to list.
    lst = to_list(lst)

    # Process elements.
    return [process_elem(i, **kwargs) for i in lst]


def process_column(col: Column, **kwargs: bool) -> Column:
    """Preprocess a Column object.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        A column to preprocess.
    **kwargs : dict, optional
        Preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        The processed column.

    """
    # Extract kwargs.
    lower = kwargs.get("lower", False)
    trim = kwargs.get("trim", False)
    to_str = kwargs.get("to_str", False)
    to_int = kwargs.get("to_int", False)
    to_float = kwargs.get("to_float", False)
    to_bool = kwargs.get("to_bool", False)
    to_date = kwargs.get("to_date", False)
    to_timestamp = kwargs.get("to_timestamp", False)

    # Convert to string.
    if to_str:
        col = cast(col, String)

    # If a string column.
    if "VARCHAR" in str(col.type):
        # Lower column.
        if lower:
            col = func.lower(col)

        # Trim whitespace.
        if trim:
            col = func.trim(col)

    if to_int:
        col = cast(col, Integer)

    if to_float:
        col = cast(col, Float)

    if to_bool:
        col = cast(col, Boolean)

    if to_date:
        col = cast(col, DATE)

    if to_timestamp:
        col = cast(col, TIMESTAMP)

    return col


def equals(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column has some value.

    Assumes that if searching for a string, both the value and column values
    should be converted to lowercase and trimmed of leading/trailing whitespace.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    val : Any
        The value to match in the column.
    lower : bool, default=True
        Whether to convert the value and column to lowercase.
        This is only relevant when the column/value are strings.
    trim : bool, default=True
        Whether to trim (strip) whitespace on the value and column.
        This is only relevant when the column/value are strings.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return process_column(col, lower=lower, trim=trim, **kwargs) == process_elem(
        value, lower=lower, trim=trim, **kwargs
    )


def not_equals(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column is not equal to some value.

    Assumes that if searching for a string, both the value and column values
    should be converted to lowercase and trimmed of leading/trailing whitespace.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    val : Any
        The value to match in the column.
    lower : bool, default=True
        Whether to convert the value and column to lowercase.
        This is only relevant when the column/value are strings.
    trim : bool, default=True
        Whether to trim (strip) whitespace on the value and column.
        This is only relevant when the column/value are strings.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return process_column(col, lower=lower, trim=trim, **kwargs) != process_elem(
        value, lower=lower, trim=trim, **kwargs
    )


def has_string_format(
    col: Column, value: Any, fmt: str, to_str: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column has some string formatting.

    Assumes that we're searching for a string, performing
    the relevant conversion.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    value: Any
        A value to be implanted in the string formatting.
    fmt : str
        The string format to match in the column.
    to_str : bool, default=True
        Whether to convert the value/column to string type.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return process_column(col, to_str=to_str, **kwargs).like(
        fmt.format(process_elem(value, to_str=to_str, **kwargs))
    )


def has_substring(
    col: Column, substring: Any, lower: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column has some substring.

    Assumes that we're searching for a string, where both the value and
    column values should be converted to strings and made lowercase.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    substring : Any
        The substring to match in the column.
    lower : bool, default=True
        Whether to convert the value and column to lowercase.
        This is only relevant when the column/value are strings.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return has_string_format(col, substring, "%%{}%%", lower=lower, **kwargs)


def starts_with(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column starts with some value/string.

    Assumes that we're searching for a string, where both the value and
    column values should be converted to strings, made lowercase, and
    trimmed of leading/trailing whitespace.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    value : Any
        The value to match at the start.
    lower : bool, default=True
        Whether to convert the value and column to lowercase.
        This is only relevant when the column/value are strings.
    trim : bool, default=True
        Whether to trim (strip) whitespace on the value and column.
        This is only relevant when the column/value are strings.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return has_string_format(col, value, "{}%%", lower=lower, trim=trim, **kwargs)


def ends_with(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column ends with some value/string.

    Assumes that we're searching for a string, where both the value and
    column values should be converted to strings, made lowercase, and
    trimmed of leading/trailing whitespace.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    value : Any
        The value to match at the end.
    lower : bool, default=True
        Whether to convert the value and column to lowercase.
        This is only relevant when the column/value are strings.
    trim : bool, default=True
        Whether to trim (strip) whitespace on the value and column.
        This is only relevant when the column/value are strings.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return has_string_format(col, value, "%%{}", lower=lower, trim=trim, **kwargs)


def in_(
    col: Column, lst: List[Any], lower: bool = True, trim: bool = True, **kwargs: bool
) -> BinaryExpression:
    """Condition that a column value is in a list of values.

    Assumes that if searching for a string, both the value and column values
    should be converted to lowercase and trimmed of leading/trailing whitespace.

    Parameters
    ----------
    col : sqlalchemy.sql.schema.Column
        The column to condition.
    lst : list of any
        The value to match at the start.
    lower : bool, default=True
        Whether to convert the value and column to lowercase.
        This is only relevant when the column/value are strings.
    trim : bool, default=True
        Whether to trim (strip) whitespace on the value and column.
        This is only relevant when the column/value are strings.
    **kwargs : dict, optional
        Remaining preprocessing keyword arguments.

    Returns
    -------
    sqlalchemy.sql.elements.BinaryExpression
        An expression representing where the condition was satisfied.

    """
    return process_column(col, lower=lower, trim=trim, **kwargs).in_(
        process_list(lst, lower=lower, trim=trim, **kwargs)
    )


def check_column_type(
    table: TableTypes,
    cols: Union[str, List[str]],
    types: Union[Any, List[Any]],
    raise_error=False,
):
    """Check whether some columns are each one of a number of types.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    cols: str or list of str
        Column names to check.
    types: any
        The allowed types for each column.
    raise_error: bool
        Whether to raise an error if one of the columns are none of the types.

    Returns
    -------
    bool
        Whether all of the columns are one of the types.

    """
    cols = to_list(cols)
    types = to_list(types)
    is_type = [
        any(isinstance(get_column(table, col).type, type_) for type_ in types)
        for col in cols
    ]

    if raise_error and not all(is_type):
        incorrect_type = list(
            set(cols) - {col for i, col in enumerate(cols) if is_type[i]}
        )
        types_str = ", ".join([type_.__name__ for type_ in types])
        actual_types_str = [type(col).__name__ for col in incorrect_type]
        raise ValueError(
            f"""{incorrect_type} columns are not one of types {types_str}.
            They have types {actual_types_str}."""
        )

    return all(is_type)


def check_timestamp_columns(
    table: TableTypes, cols: Union[str, List[str]], raise_error=False
):
    """Check whether some columns are DATE or TIMESTAMP columns.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    cols: str or list of str
        Column names to check.
    raise_error: bool
        Whether to raise an error if one of the columns are none of the types.

    Returns
    -------
    bool
        Whether all of the columns are one of the types.

    """
    return check_column_type(table, cols, [DATE, TIMESTAMP], raise_error=raise_error)


@table_params_to_type(Subquery)
def get_delta_column(
    table: TableTypes,
    years: Optional[str] = None,
    months: Optional[str] = None,
    days: Optional[str] = None,
    hours: Optional[str] = None,
) -> Column:
    """Create a time delta column.

    Create a time delta (interval) column from a number of
    numeric timestamp columns.

    Warning: Null values in each specified numeric time column are coalesced to 0.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
        The table.
    years: None or str
        Years column.
    months: None or str
        Months column.
    days: None or str
        Days column.
    hours: None or str
        Hours column.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        Combined delta/interval column.

    """

    def get_col_or_none(col):
        """If col is not None, get interval column from names."""
        return None if col is None else get_column(table, col)

    years = get_col_or_none(years)
    months = get_col_or_none(months)
    days = get_col_or_none(days)
    hours = get_col_or_none(hours)

    time_cols = [years, months, days, hours]
    names = ["YEARS", "MONTHS", "DAYS", "HOURS"]

    # Consider only the non-null columns.
    names = [names[i] for i in range(len(names)) if time_cols[i] is not None]
    time_cols = [col for col in time_cols if col is not None]

    if len(time_cols) == 0:
        raise ValueError("One or more time interval columns must be specified.")

    # Create interval columns.
    interval_cols = []
    for i, col in enumerate(time_cols):
        interval_cols.append(
            func.cast(func.concat(func.coalesce(col, 0), " " + names[i]), Interval)
        )

    # Create combined interval column.
    combined_interval_col = interval_cols[0]
    for i in range(1, len(interval_cols)):
        combined_interval_col = combined_interval_col + interval_cols[i]

    return combined_interval_col
