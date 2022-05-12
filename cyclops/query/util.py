"""Utility functions for querying."""

# mypy: ignore-errors
# pylint: disable=too-many-lines

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, List, Union

import numpy as np
import sqlalchemy
from sqlalchemy import Float, Integer, Interval, String, cast, func, select
from sqlalchemy.dialects.postgresql.base import TIMESTAMP
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import ColumnClause
from sqlalchemy.sql.schema import Column, Table
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)

COLUMN_TYPES = [Column, ColumnClause]


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


class DBMetaclass(type):
    """Meta class for Database, keeps track of instances for singleton."""

    __instances: dict = {}

    def __call__(cls, *args, **kwargs):
        """Call."""
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)

        return cls.__instances[cls]


QUERY_TYPES = [Table, Select, Subquery, DBTable]


def to_list(obj: Any) -> list:
    """Convert some object to a list of object(s) unless already one.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed function.

    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, np.ndarray):
        return list(obj)

    return [obj]


def to_datetime_format(date: str) -> datetime:
    """Convert date in string format YYYY-MM-DD to datetime.

    Parameters
    ----------
    date: str
        Input date in string format.
    Returns
    -------
    datetime
        Date in datetime format.

    """
    return datetime.strptime(date, "%Y-%m-%d")


def _to_subquery(table_: Union[Select, Subquery, Table, DBTable]) -> Subquery:
    """Convert a query from some type in QUERY_TYPES to Subquery type.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        Query to convert.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The converted query.

    """
    if isinstance(table_, Subquery):
        return table_

    if isinstance(table_, Select):
        return table_.subquery()

    if isinstance(table_, Table):
        return select(table_).subquery()

    if isinstance(table_, DBTable):
        return select(table_.data).subquery()

    raise ValueError(
        f"""table_ has type {type(table_)}, but must have one of the
        following types: {", ".join(QUERY_TYPES)}"""
    )


def _to_select(table_: Union[Select, Subquery, Table, DBTable]) -> Select:
    """Convert a query from some type in QUERY_TYPES to Select type.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        Query to convert.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The converted query.

    """
    if isinstance(table_, Select):
        return table_

    if isinstance(table_, Subquery):
        return select(table_)

    if isinstance(table_, Table):
        return select(table_)

    if isinstance(table_, DBTable):
        return select(table_.data)

    raise ValueError(
        f"""t has type {type(table_)}, but must have one of the
        following types: {", ".join(QUERY_TYPES)}"""
    )


def param_types_to_type(relevant_types: List[Any], to_type_fn: Callable) -> Callable:
    """Decorate function with conversion of input types to a specific type.

    Decorator which processes a function's arguments by taking all
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
        """Decorate function to convert query type parameters to Subquery type."""

        @wraps(func_)
        def wrapper_func(*args, **kwargs):
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


def query_params_to_type(to_type: Union[Table, Select, Subquery, DBTable]) -> Callable:
    """Decorate function to convert query type params to specified type in QUERY_TYPES.

    Parameters
    ----------
    to_type: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The type to which to convert.

    Returns
    -------
    Callable
        The processed function.

    """
    # Dictionary mapping query type -> query type conversion function.
    query_to_type_fn_map = {
        Subquery: _to_subquery,
        Select: _to_select,
        Table: lambda x: x,
        DBTable: lambda x: x,
    }
    if to_type not in QUERY_TYPES:
        raise ValueError(f"to_type must be in {QUERY_TYPES}")

    to_type_fn = query_to_type_fn_map[to_type]

    return param_types_to_type(QUERY_TYPES, to_type_fn)


@query_params_to_type(Subquery)
def get_attribute(
    table_: Union[Select, Subquery, Table, DBTable],
    attr: Union[str, Column],
    assert_same: bool = True,
):
    """Extract an attribute object from the subquery.

    The attribute given may be a column object or the column name (string).
    The assert_same parameter is designed as a fail-safe such that
    when a column object is passed in, it is asserted to be in the
    query. Made optional because an attribute may have been altered.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query with the column.
    attr: str or sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.
    assert_same : bool
        Whether to assert that a Column object is the same one as in the query.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        The corresponding attribute in the query.

    """
    col_names = [c.name for c in table_.columns]

    # If a Column object.
    if type(attr) in COLUMN_TYPES:
        # Make sure the column object provided matches
        # the column in the query of the same name.
        if assert_same:
            if attr != table_.c[col_names.index(attr.name)]:
                raise ValueError(
                    """Column provided and that of the same name
                    in the query do not match. This assertion can be ignored by
                    setting assert_same=False."""
                )
        return attr

    # Otherwise, a string.
    if attr not in col_names:
        raise ValueError(f"Query does not contain column {attr}")

    return table_.c[col_names.index(attr)]


@query_params_to_type(Subquery)
def filter_attributes(
    table_: Union[Select, Subquery, Table, DBTable],
    attrs: Union[str, Column, List[Union[str, Column]]],
) -> Subquery:
    """Filter a number of attributes from the subquery.

    The attribute(s) may be a column name (string),
    or a list of any strings.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query with the column.
    attrs: str or sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Filtered attributes from the query as a new subquery.

    """
    col_names = [c.name for c in table_.columns]
    filtered = []
    for attr in to_list(attrs):
        if attr not in col_names:
            continue
        filtered.append(table_.c[col_names.index(attr)])

    return select(filtered).subquery()


@query_params_to_type(Subquery)
def get_attributes(
    table_: Union[Select, Subquery, Table, DBTable],
    attrs: Union[str, Column, List[Union[str, Column]]],
    assert_same: bool = True,
):
    """Extract a number of attributes from the subquery.

    The attribute(s) given may be a column object, column name
    (string), or a list of any combination of these objects.

    The assert_same parameter is designed as a fail-safe such that
    when a column object is passed in, it is asserted to be in the
    query. Made optional because an attribute may have been altered.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query with the column.
    attrs: str or sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.
    assert_same : bool
        Whether to assert that a Column object is the same one as in the query.

    Returns
    -------
    list of sqlalchemy.sql.schema.Column
        The corresponding attributes in the query.

    """
    return [
        get_attribute(table_, attr, assert_same=assert_same) for attr in to_list(attrs)
    ]


@query_params_to_type(Subquery)
def drop_attributes(
    table_: Union[Select, Subquery, Table, DBTable],
    drop_cols: Union[str, Column, List[Union[str, Column]]],
) -> Subquery:
    """Drop some attribute(s) from a query.

    The attribute(s) given may be a column object, column name (string), or a
    list of any combination of these objects.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query with the column.
    col : str or sqlalchemy.sql.schema.Column or list of str or list of
    sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The corresponding query with attributes dropped.

    """
    drop_cols = get_attributes(table_, drop_cols)

    return select(*[c for c in table_.c if c not in drop_cols]).subquery()


@query_params_to_type(Subquery)
def rename_attributes(
    table_: Union[Select, Subquery, Table, DBTable], old_new_map: dict
) -> Subquery:
    """Rename a query's attributes.

    Rename query's attributes according to a dictionary of strings,
    where the key is the current name, and the value is the replacement.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    d : dict
        Dictionary mapping current attribute names (key) to new ones (value).

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The corresponding query with attributes renamed.

    """
    return select(
        *[
            c.label(old_new_map[c.name]) if c.name in old_new_map else c
            for c in table_.c
        ]
    ).subquery()


@query_params_to_type(Subquery)
def reorder_attributes(
    table_: Union[Select, Subquery, Table, DBTable],
    cols: List[Union[str, Column]],
    assert_same: bool = True,
) -> Subquery:
    """Reorder a query's attributes.

    Reorder query's attributes according to a list of strings or
    column objects in the query.

    The assert_same parameter is designed as a fail-safe such that
    when a column object is passed in, it is asserted to be in the
    query. Made optional because an attribute may have been altered.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query to reorder.
    cols : list of str or list of sqlalchemy.sql.schema.Column
        New attribute order.
    assert_same : bool
        Whether to assert that a Column object is the same one as in the query.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The reordered query.

    """
    # Get the old/new column names.
    old_order = [c.name for c in table_.c]
    new_order = [c.name for c in get_attributes(table_, cols, assert_same=assert_same)]

    # Make sure we have exactly the same set of old/new column names.
    if not set(old_order) == set(new_order):
        old_order_print = ", ".join(old_order)
        new_order_print = ", ".join(new_order)
        raise ValueError(
            f"""Must specify all the query's attributes {old_order_print}
            to re-order, not {new_order_print}."""
        )

    # Reorder the columns.
    new_cols = []
    for col in new_order:
        new_cols.append(table_.c[old_order.index(col)])

    return select(*new_cols).subquery()


@query_params_to_type(Subquery)
def apply_to_attributes(
    table_: Union[Select, Subquery, Table, DBTable],
    cols: List,
    func_: Callable,
) -> Subquery:
    """Apply a function to some attributes.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    cols : list of str or list of sqlalchemy.sql.schema.Column
        Attributes to which to apply the function.
    func_: Callable
        Function to apply to the attributes, where it takes an attribute
        as its only parameter and returns another column object.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The query with function applied.

    """
    org_col_names = list(table_.c)
    cols = get_attributes(table_, cols)
    if isinstance(cols[0], Column):
        col_names = [c.name for c in cols]
    if isinstance(cols[0], str):
        col_names = cols

    # Apply function to columns.
    name_d = {}
    trimmed = []
    for col in table_.c:
        if col not in cols:
            continue
        new_name = "temp" + col.name + "temp"
        name_d[new_name] = col.name
        trimmed.append(func_(col).label("temp" + col.name + "temp"))

    # Append new columns.
    subquery = select(table_, *trimmed).subquery()

    # Drop old columns.
    subquery = drop_attributes(subquery, col_names)

    # Rename those temporary columns.
    subquery = rename_attributes(subquery, name_d)

    # Re-order the columns as they were originally.
    query = reorder_attributes(subquery, org_col_names, assert_same=False)

    return query


def trim_attributes(
    table_: Union[Select, Subquery, Table, DBTable],
    cols: List[Union[str, Column]],
) -> Subquery:
    """Trim attributes and remove leading/trailing whitespace.

    Returns query with columns listed having their
    leading/trailing whitespace trimmed (stripped).

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    cols : list of str or list of sqlalchemy.sql.schema.Column
        Attributes to trim.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The query with trimmed attrbutes.

    """
    return apply_to_attributes(
        table_, cols, lambda x: process_attribute(x, to_str=True, trim=True)
    )


def rga(obj, *attr_args):
    """Recursive getattr (rga): express a series of attribute accesses with strings.

    E.g., obj.a.b.c == rga(obj, "a", "b", "c")

    Parameters
    ----------
    oobj: any
        Inital object.
    *attr_args : list of str
        Ordered list of attributes to access.

    Returns
    -------
    any
        The object accessed by the final attribute.

    """
    # Get attribute.
    next_attr = getattr(obj, attr_args[0])

    # Base case.
    if len(attr_args) == 1:
        return next_attr

    # Recurse.
    return getattr(next_attr, attr_args[1:])


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

    # Convert to string.
    if to_str:
        elem = str(elem)

    # If a string.
    if isinstance(elem, str):
        if lower:
            elem = elem.lower()

        if trim:
            elem = elem.strip()

    # Convert to int.
    if to_int:
        elem = int(elem)

    # Convert to float.
    if to_float:
        elem = float(elem)

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


def process_attribute(col: Column, **kwargs: bool) -> Column:
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
        The preprocessed column.

    """
    # Extract kwargs.
    lower = kwargs.get("lower", False)
    trim = kwargs.get("trim", False)
    to_str = kwargs.get("to_str", False)
    to_int = kwargs.get("to_int", False)
    to_float = kwargs.get("to_float", False)

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
    return process_attribute(col, lower=lower, trim=trim, **kwargs) == process_elem(
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
    return process_attribute(col, lower=lower, trim=trim, **kwargs) != process_elem(
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
    return process_attribute(col, to_str=to_str, **kwargs).like(
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
    return process_attribute(col, lower=lower, trim=trim, **kwargs).in_(
        process_list(lst, lower=lower, trim=trim, **kwargs)
    )


def create_interval_attribute(
    years: Union[None, Column] = None,
    months: Union[None, Column] = None,
    days: Union[None, Column] = None,
    hours: Union[None, Column] = None,
) -> Column:
    """Create an interval type column from a number of time columns.

    Warning: Null values in each specified time column are coalesced to 0.

    Parameters
    ----------
    years: None or sqlalchemy.sql.schema.Column
        Years column.
    months: None or sqlalchemy.sql.schema.Column
        Months column.
    days: None or sqlalchemy.sql.schema.Column
        Days column.
    hours: None or sqlalchemy.sql.schema.Column
        hours column.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        Combined interval column.

    """
    time_cols = [years, months, days, hours]
    names = ["YEARS", "MONTHS", "DAYS", "HOURS"]

    # Consider only the non-null columns.
    names = [names[i] for i in range(len(names)) if time_cols[i] is not None]
    time_cols = [col for col in time_cols if col is not None]

    if len(time_cols) == 0:
        raise ValueError("One or more time columns must be specified.")

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


def add_interval_attribute_to_table(
    table_: Union[Select, Subquery, Table, DBTable],
    years: Union[None, str] = None,
    months: Union[None, str] = None,
    days: Union[None, str] = None,
    hours: Union[None, str] = None,
):
    """Create an interval type column and add this attribute to the provided table.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query containing the time columns.
    years : None or str
        Attribute name for years column in table_.
    months : None or str
        Attribute name for months column in table_.
    days : None or str
        Attribute name for days column in table_.
    hours : None or str
        Attribute name for hours column in table_.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The query with the interval.
    Column
        Interval column.

    """
    # If not None, get interval column from names.
    def get_attr_or_none(col):
        return None if col is None else get_attribute(table_, col)

    interval_col = create_interval_attribute(
        years=get_attr_or_none(years),
        months=get_attr_or_none(months),
        days=get_attr_or_none(days),
        hours=get_attr_or_none(hours),
    )

    query = select(table_, interval_col).subquery()

    return query, interval_col


def add_interval_to_timestamps(  # pylint:disable=too-many-arguments
    table_: Union[Select, Subquery, Table, DBTable],
    timestamp_cols: Union[str, List[str]],
    years: Union[None, str] = None,
    months: Union[None, str] = None,
    days: Union[None, str] = None,
    hours: Union[None, str] = None,
) -> Subquery:
    """Create and add an interval column and adds it to a table.

    The individual time columns are given as column names in the table.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query containing the timestamp_cols and time columns.
    years : None or str
        Attribute name for years column in table_.
    months : None or str
        Attribute name for months column in table_.
    days : None or str
        Attribute name for years column in table_.
    hours : None or str
        Attribute name for years column in table_.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The query with the modified timestamp_cols.

    """
    # Ensure timestamp columns are actually timestamps.
    for col in to_list(timestamp_cols):
        if not isinstance(get_attribute(table_, col).type, TIMESTAMP):
            raise ValueError(f"{col} must be a timestamp column.")

    # Get combined interval column.
    _, interval_col = add_interval_attribute_to_table(
        table_,
        years=years,
        months=months,
        days=days,
        hours=hours,
    )

    return apply_to_attributes(table_, timestamp_cols, lambda x: x + interval_col)
