"""Utility functions for querying."""

# mypy: ignore-errors
# pylint: disable=too-many-lines

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, List, Optional, Union

import numpy as np
import sqlalchemy
from sqlalchemy import Float, Integer, Interval, String, cast, func, select
from sqlalchemy.dialects.postgresql.base import DATE, TIMESTAMP
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.expression import ColumnClause
from sqlalchemy.sql.schema import Column, Table
from sqlalchemy.sql.selectable import Select, Subquery

from codebase_ops import get_log_file_path
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)

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


class DBMetaclass(type):
    """Meta class for Database, keeps track of instances for singleton."""

    __instances: dict = {}

    def __call__(cls, *args, **kwargs):
        """Call."""
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)

        return cls.__instances[cls]


QUERY_OBJECTS = [Table, Select, Subquery, DBTable]
QueryTypes = Union[Select, Subquery, Table, DBTable]


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


def to_list_optional(obj: Optional[Any]) -> Optional[list]:
    """Convert some object to a list of object(s) unless already None or a list.

    Parameters
    ----------
    obj : any
        The object to convert to a list.

    Returns
    -------
    list
        The processed function.

    """
    if obj is None:
        return None

    if isinstance(obj, list):
        return obj

    if isinstance(obj, np.ndarray):
        return list(obj)

    return [obj]


def to_datetime_format(date: str, fmt="%Y-%m-%d") -> datetime:
    """Convert string date to datetime.

    Parameters
    ----------
    date: str
        Input date in string format.
    fmt: str, optional
        Date formatting string.

    Returns
    -------
    datetime
        Date in datetime format.

    """
    return datetime.strptime(date, fmt)


def _to_subquery(table_: QueryTypes) -> Subquery:
    """Convert a query from some type in QUERY_OBJECTS to Subquery type.

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
        following types: {", ".join(QUERY_OBJECTS)}"""
    )


def _to_select(table_: QueryTypes) -> Select:
    """Convert a query from some type in QUERY_OBJECTS to Select type.

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
        following types: {", ".join(QUERY_OBJECTS)}"""
    )


def param_types_to_type(relevant_types: List[Any], to_type_fn: Callable) -> Callable:
    """Convert QueryType parameters to a specified type.

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
        """Decorate function to convert QueryType parameters to a specified type."""

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


def query_params_to_type(to_type: QueryTypes) -> Callable:
    """Decorate to convert QueryTypes params to a specified type.

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
    if to_type not in QUERY_OBJECTS:
        raise ValueError(f"to_type must be in {QUERY_OBJECTS}")

    to_type_fn = query_to_type_fn_map[to_type]

    return param_types_to_type(QUERY_OBJECTS, to_type_fn)


@query_params_to_type(Subquery)
def has_attributes(
    table_: QueryTypes, attrs: Union[str, List[str]], raise_error: bool = False
):
    """Check whether a table has all of the given columns.

    Parameters
    ----------
    table_ : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        Table to check.
    attrs: str or list of str
        Required columns.
    raise_error: bool
        Whether to raise an error if the required columns are not found.

    """
    attrs = to_list(attrs)
    table_cols = get_attribute_names(table_)
    attrs_in_table = [attr in table_cols for attr in attrs]
    if raise_error and not all(attrs_in_table):
        missing = list(set(attrs) - set(attrs_in_table))
        missing_str = ", ".join(missing)
        if len(missing) == 1:
            raise ValueError(f"Column {missing_str} is not in table.")
        raise ValueError(f"{missing_str} columns are not in table.")
    return all(attrs_in_table)


@query_params_to_type(Subquery)
def get_attribute(
    table_: QueryTypes,
    attr: str,
):
    """Extract an attribute object from the subquery.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query with the column.
    attr: str
        Name of attribute to extract.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        The corresponding attribute in the query.

    """
    col_names = get_attribute_names(table_)
    if attr not in col_names:
        raise ValueError(f"Query does not contain column {attr}")

    return table_.c[col_names.index(attr)]


@query_params_to_type(Subquery)
def filter_attributes(
    table_: QueryTypes,
    attrs: Union[str, List[str]],
) -> Subquery:
    """Filter a table, keeping only the specified columns.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query with the column.
    attrs: str or list of str
        Name of attribute on which to filter.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        Filtered attributes from the query as a new subquery.

    """
    attrs = to_list(attrs)
    if len(attrs) == 0:
        raise ValueError("Must specify at least one column to filter.")

    col_names = get_attribute_names(table_)
    filtered = []
    for attr in attrs:
        if attr not in col_names:
            continue
        filtered.append(table_.c[col_names.index(attr)])

    return select(filtered).subquery()


@query_params_to_type(Subquery)
def get_attributes(
    table_: QueryTypes,
    attrs: Union[str, List[str]],
):
    """Extract a number of attributes from the subquery.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The table.
    attrs: str or list of str
        Names of attributes to extract.

    Returns
    -------
    list of sqlalchemy.sql.schema.Column
        The corresponding attributes in the query.

    """
    return [get_attribute(table_, attr) for attr in to_list(attrs)]


@query_params_to_type(Subquery)
def get_attribute_names(table_: QueryTypes):
    """Extract attribute names from a table.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The table.

    Returns
    -------
    list of str
        The table attribute names.

    """
    return [c.name for c in table_.columns]


@query_params_to_type(Subquery)
def has_columns(
    table_: QueryTypes, cols: Union[str, List[str]], raise_error: bool = False
) -> bool:
    """Check if data has required columns for processing.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The table.
    cols: str or list of str
        List of column names that must be present in data.
    raise_error: bool
        Whether to raise a ValueError if there are missing columns.

    Returns
    -------
    bool
        True if all required columns are present, otherwise False.

    """
    cols = to_list(cols)
    required_set = set(cols)
    columns = set(get_attribute_names(table_))
    present = required_set.issubset(columns)

    if raise_error and not present:
        missing = required_set - columns
        raise ValueError(f"Missing required columns {missing}")

    return present


@query_params_to_type(Subquery)
def assert_query_has_columns(*args, **kwargs) -> Callable:
    """Assert that QueryType params have the necessary columns.

    assert_query_has_columns(["A", "B"], None) is equivalent to
    assert_query_has_columns(["A", "B"]) but may be necessary when
    wanting to check, assert_query_has_columns(["A"], None, ["C"])

    Can also check keyword arguments, e.g., optional queries,
    assert_query_has_columns(["A"], optional_query=["D"])

    Parameters
    ----------
    *args
        Required columns of the function's ordered query arguments.
    **kwargs
        Keyword corresponds to the query kwargs of the function.
        The value is this keyword argument's required columns.

    Returns
    -------
    Callable
        Decorator function.

    """

    def decorator(func_: Callable) -> Callable:
        @wraps(func_)
        def wrapper_func(*fn_args, **fn_kwargs) -> Callable:
            # Check only the query arguments
            query_args = [i for i in fn_args if isinstance(i, Subquery)]

            assert len(args) <= len(query_args)

            for i, arg in enumerate(args):
                if arg is None:  # Can specify None to skip over checking a query
                    continue
                has_columns(query_args[i], arg, raise_error=True)

            for key, required_cols in kwargs.items():
                # If an optional query is not provided, or is None,
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


@query_params_to_type(Subquery)
def drop_attributes(
    table_: QueryTypes,
    drop_cols: Union[str, List[str]],
) -> Subquery:
    """Drop some attribute(s) from a query.

    The attribute(s) given may be a column object, column name (string), or a
    list of any combination of these objects.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The table.
    col : str or list of str
        Names of attributes to drop (remove).

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The corresponding query with attributes dropped.

    """
    drop_cols = get_attributes(table_, drop_cols)

    return select(*[c for c in table_.c if c not in drop_cols]).subquery()


@query_params_to_type(Subquery)
def rename_attributes(table_: QueryTypes, rename_map: dict) -> Subquery:
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
            c.label(rename_map[c.name]) if c.name in rename_map else c
            for c in table_.columns
        ]
    ).subquery()


@query_params_to_type(Subquery)
def reorder_attributes(table_: QueryTypes, cols: List[str]) -> Subquery:
    """Reorder a query's attributes.

    Reorder query's attributes according to a list of strings or
    column objects in the query.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query to reorder.
    cols : list of str
        New attribute order.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The reordered query.

    """
    # Get the old/new column names.
    old_order = get_attribute_names(table_)
    new_order = [c.name for c in get_attributes(table_, cols)]

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
    table_: QueryTypes,
    col_names: Union[str, List[str]],
    func_: Callable,
    new_col_labels: Optional[Union[str, List[str]]] = None,
) -> Subquery:
    """Apply a function to some attributes.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    col_names: str or list of str
        Attributes to which to apply the function.
    func_: Callable
        Function to apply to the attributes, where it takes an attribute
        as its only parameter and returns another column object.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The query with function applied.

    """
    col_names = to_list(col_names)
    new_col_labels = to_list_optional(new_col_labels)
    cols = get_attributes(table_, col_names)

    if new_col_labels is None:
        # Apply to existing attributes
        prev_order = get_attribute_names(table_)
        table_ = select(table_).add_columns(
            *[
                func_(col).label("__" + col_names[i] + "__")
                for i, col in enumerate(cols)
            ]
        )
        rename = {"__" + name + "__": name for name in col_names}
        table_ = drop_attributes(table_, col_names)
        table_ = rename_attributes(table_, rename)
        table_ = reorder_attributes(table_, prev_order)
    else:
        # Apply to new attributes
        new_cols = [func_(col).label(new_col_labels[i]) for i, col in enumerate(cols)]
        table_ = select(table_).add_columns(*new_cols).subquery()

    return _to_subquery(table_)


def trim_attributes(
    table_: QueryTypes,
    cols: Union[str, List[str]],
    new_col_labels: Optional[Union[str, List[str]]] = None,
) -> Subquery:
    """Trim attributes and remove leading/trailing whitespace.

    Returns query with columns listed having their
    leading/trailing whitespace trimmed (stripped).

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    cols: str or list of str
        Names of attributes to trim.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The query with trimmed attrbutes.

    """
    return apply_to_attributes(
        table_,
        cols,
        lambda x: process_attribute(x, to_str=True, trim=True),
        new_col_labels=new_col_labels,
    )


def rga(obj, *attr_args):
    """Recursive getattr (rga): express a series of attribute accesses with strings.

    E.g., obj.a.b.c == rga(obj, "a", "b", "c")

    Parameters
    ----------
    obj: any
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
        The processed column.

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


def check_attribute_type(
    table_: QueryTypes,
    attrs: Union[str, List[str]],
    types: Union[Any, List[Any]],
    raise_error=False,
):
    """Check whether some columns are each one of a number of types.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    attrs: str or list of str
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
    attrs = to_list(attrs)
    types = to_list(types)
    is_type = [
        any(isinstance(get_attribute(table_, attr).type, type_) for type_ in types)
        for attr in attrs
    ]

    if raise_error and not all(is_type):
        incorrect_type = set(attrs) - {
            attr for i, attr in enumerate(attrs) if is_type[i]
        }
        types_str = ", ".join([type_.__name__ for type_ in types])
        raise ValueError(
            f"{incorrect_type} attributes are not one of types {types_str}."
        )

    return all(is_type)


def check_timestamp_attributes(
    table_: QueryTypes, attrs: Union[str, List[str]], raise_error=False
):
    """Check whether some columns are DATE or TIMESTAMP columns.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
    attrs: str or list of str
        Column names to check.
    raise_error: bool
        Whether to raise an error if one of the columns are none of the types.

    Returns
    -------
    bool
        Whether all of the columns are one of the types.

    """
    return check_attribute_type(
        table_, attrs, [DATE, TIMESTAMP], raise_error=raise_error
    )


@query_params_to_type(Subquery)
def get_delta_attribute(
    table_: QueryTypes,
    years: Optional[str] = None,
    months: Optional[str] = None,
    days: Optional[str] = None,
    hours: Optional[str] = None,
) -> Column:
    """Create a time delta attribute.

    Create a time delta (interval) attribute from a number of
    numeric timestamp columns.

    Warning: Null values in each specified numeric time column are coalesced to 0.

    Parameters
    ----------
    table_: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        The query.
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

    def get_attr_or_none(col):
        """If col is not None, get interval column from names."""
        return None if col is None else get_attribute(table_, col)

    years = get_attr_or_none(years)
    months = get_attr_or_none(months)
    days = get_attr_or_none(days)
    hours = get_attr_or_none(hours)

    time_cols = [years, months, days, hours]
    names = ["YEARS", "MONTHS", "DAYS", "HOURS"]

    # Consider only the non-null columns.
    names = [names[i] for i in range(len(names)) if time_cols[i] is not None]
    time_cols = [col for col in time_cols if col is not None]

    if len(time_cols) == 0:
        raise ValueError("One or more time interval attributes must be specified.")

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
