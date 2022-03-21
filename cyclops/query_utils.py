"""Utility functions for querying."""

from typing import List, Any, Callable, Union

import numpy as np
import sqlalchemy
from sqlalchemy import cast
from sqlalchemy import String, Integer, Float
from sqlalchemy import select, func
from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.sql.schema import Table, Column
from sqlalchemy.sql.elements import BinaryExpression

from dataclasses import dataclass

import logging
from cyclops.utils.log import setup_logging
from codebase_ops import get_log_file_path

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class DBSchema:
    """Database schema wrapper.

    Attributes
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

    Attributes
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


# === GENERAL PURPOSE FUNCTIONS ===
def debug_query_msg(func: Callable) -> Callable:
    """Debug message decorator function.

    Parameters
    ----------
    func: function
        Function to apply decorator.

    Returns
    -------
    Callable
        Wrapper function to apply as decorator.
    """

    def wrapper_func(*args, **kwargs):
        LOGGER.debug("Running query function: %s", {func.__name__})
        query_result = func(*args, **kwargs)
        LOGGER.debug("Finished query function: %s", {func.__name__})
        return query_result

    return wrapper_func


def to_list(a: Any):
    """Converts some object to a list object unless already one.

    Parameters
    ----------
    a : any
        The object to convert to a list.

    Returns
    -------
    Callable
        The processed function.
    """
    if isinstance(a, list):
        return a

    if isinstance(a, np.ndarray):
        a = list(a)

    return [a]


# === TYPE/QUERY CONVERSION ===
def _to_subquery(t: Union[Select, Subquery, Table, DBTable]) -> Subquery:
    """Converts a query from some type in QUERY_TYPES to
    type sqlalchemy.sql.selectable.Subquery.

    Parameters
    ----------
    t: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table
        Query to convert.

    Returns
    -------
    sqlalchemy.sql.selectable.Subquery
        The converted query.
    """
    if isinstance(t, Subquery):
        return t

    elif isinstance(t, Select):
        return t.subquery()

    elif isinstance(t, Table):
        return select(t).subquery()

    elif isinstance(t, DBTable):
        return select(t.data).subquery()

    raise ValueError(
        """t has type {}, but must have one of the
        following types: {}""".format(
            type(t), ", ".join(QUERY_TYPES)
        )
    )


def _to_select(t: Union[Select, Subquery, Table, DBTable]) -> Select:
    """Converts a query from some type in QUERY_TYPES to
    type sqlalchemy.sql.selectable.Select.

    Parameters
    ----------
    t: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table
        Query to convert.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The converted query.
    """
    if isinstance(t, Select):
        return t

    elif isinstance(t, Subquery):
        return select(t)

    elif isinstance(t, Table):
        return select(t)

    elif isinstance(t, DBTable):
        return select(t.data)

    raise ValueError(
        """t has type {}, but must have one of the
        following types: {}""".format(
            type(t), ", ".join(QUERY_TYPES)
        )
    )


# Dictionary mapping query type -> query type conversion function
QUERY_TO_TYPE_FNS = {
    Subquery: _to_subquery,
    Select: _to_select,
    Table: lambda x: x,
    DBTable: lambda x: x,
}


def param_types_to_type(relevant_types: List[Any], to_type_fn: Callable) -> Callable:
    """Decorator which processes a function's arguments by taking all
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

    def decorator(func: Callable) -> Callable:
        """Decorator function converting query type parameters
        to Subquery type.
        """
        LOGGER.debug("H0")

        def wrapper_func(*args, **kwargs):
            # Convert relevant arguments
            args = list(args)
            for i in range(len(args)):
                a = args[i]
                if type(a) in relevant_types:
                    args[i] = to_type_fn(a)

            # Convert relevant keyword arguments
            kwargs = dict(kwargs)
            for key in kwargs:
                if type(kwargs[key]) in relevant_types:
                    kwargs[key] = to_type_fn(kwargs[key])

            return func(*tuple(args), **kwargs)

        return wrapper_func

    return decorator


def query_params_to_type(to_type: Any):
    """Decorator function converting query type parameters to some
    specified query type in QUERY_TYPES.

    Parameters
    ----------
    to_type : any
        The type to which to convert.

    Returns
    -------
    Callable
        The processed function.
    """
    if to_type not in QUERY_TYPES:
        raise ValueError("to_type must be in {}".format(QUERY_TYPES))

    to_type_fn = QUERY_TO_TYPE_FNS[to_type]
    return param_types_to_type(QUERY_TYPES, to_type_fn)


# === QUERY MANIPULATION ===
@query_params_to_type(Subquery)
def get_attribute(
    t: Union[Select, Subquery, Table, DBTable],
    a: Union[str, Column],
    assert_same: bool = True,
):
    """Extracts an attribute object from the subquery. The attribute
    given may be a column object or the column name (string).

    The assert_same parameter is designed as a fail-safe such that
    when a column object is passed in, it is asserted to be in the
    query. Made optional because an attribute may have been altered.

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query with the column.
    a : str or sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.
    assert_same : bool
        Whether to assert that a Column object is the same one as in the query.

    Returns
    -------
    sqlalchemy.sql.schema.Column
        The corresponding attribute in the query.
    """
    col_names = [c.name for c in t.columns]

    # If a Column object
    if isinstance(a, Column):
        # Make sure the column object provided matches
        # the column in the query of the same name
        if assert_same:
            if a != t.c[col_names.index(a.name)]:
                raise ValueError(
                    """Column provided and that of the same name
                    in the query do not match. This assertion can be ignored by
                    setting assert_same=False."""
                )
        return a

    # Otherwise, a string
    if a not in col_names:
        raise ValueError("Query does not contain column {}".format(a))

    return t.c[col_names.index(a)]


@query_params_to_type(Subquery)
def get_attributes(
    t: Union[Select, Subquery, Table, DBTable],
    a: Union[str, Column, List[Union[str, Column]]],
    assert_same: bool = True,
):
    """Extracts a number of attributes from the subquery. The
    attribute(s) given may be a column object, column name
    (string), or a list of any combination of these objects.

    The assert_same parameter is designed as a fail-safe such that
    when a column object is passed in, it is asserted to be in the
    query. Made optional because an attribute may have been altered.

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query with the column.
    a : str or sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.
    assert_same : bool
        Whether to assert that a Column object is the same one as in the query.

    Returns
    -------
    list of sqlalchemy.sql.schema.Column
        The corresponding attributes in the query.
    """
    return [get_attribute(t, aa, assert_same=assert_same) for aa in to_list(a)]


@query_params_to_type(Subquery)
def drop_attributes(
    t: Union[Select, Subquery, Table, DBTable],
    drop_cols: Union[str, Column, List[Union[str, Column]]],
) -> Select:
    """Drops some attribute(s) from a query. The attribute(s)
    given may be a column object, column name (string), or a
    list of any combination of these objects.

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query with the column.
    col : str or sqlalchemy.sql.schema.Column or list of str or list of
    sqlalchemy.sql.schema.Column
        Attribute to get. It is either a Column object or the column name.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The corresponding query with attributes dropped.
    """
    drop_cols = get_attributes(t, drop_cols)
    return select(*[c for c in t.c if c not in drop_cols])


@query_params_to_type(Subquery)
def rename_attributes(t: Union[Select, Subquery, Table, DBTable], d: dict) -> Select:
    """Renames a query's attributes according to a dictionary of strings,
    where the key is the current name, and the value is the replacement.

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query.
    d : dict
        Dictionary mapping current attribute names (key) to new ones (value).

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The corresponding query with attributes renamed.
    """
    # Make sure we aren't renaming to columns to be the same
    d_values = list(d.values())
    if len(d_values) != len(set(d_values)):
        raise ValueError("Cannot rename two attributes to the same name.")

    # Rename
    return select(*[c.label(d[c.name]) if c.name in d else c for c in t.c])


@query_params_to_type(Subquery)
def reorder_attributes(
    t: Union[Select, Subquery, Table, DBTable],
    cols: List[Union[str, Column]],
    assert_same: bool = True,
) -> Select:
    """Reorder a query's attributes according to a list of strings or
    column objects in the query.

    The assert_same parameter is designed as a fail-safe such that
    when a column object is passed in, it is asserted to be in the
    query. Made optional because an attribute may have been altered.

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query to reorder.
    cols : list of str or list of sqlalchemy.sql.schema.Column
        New attribute order.
    assert_same : bool
        Whether to assert that a Column object is the same one as in the query.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The reordered query.
    """
    # Get the old/new column names
    old_order = [c.name for c in t.c]
    new_order = [c.name for c in get_attributes(t, cols, assert_same=assert_same)]

    # Make sure we have exactly the same set of old/new column names
    if not set(old_order) == set(new_order):
        raise ValueError(
            "Must specify all the query's attributes ({}) to re-order, not {}.".format(
                ", ".join(old_order), ", ".join(new_order)
            )
        )

    # Reorder the columns
    new_cols = []
    for c in new_order:
        new_cols.append(t.c[old_order.index(c)])

    return select(*new_cols)


@query_params_to_type(Subquery)
def apply_to_attributes(
    t: Union[Select, Subquery, Table, DBTable],
    cols: List[Union[str, Column]],
    fn: Callable,
) -> Select:
    """
    Applies a function to some attributes.

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query.
    cols : list of str or list of sqlalchemy.sql.schema.Column
        Attributes to which to apply the function.
    fn : Callable
        Function to apply to the attributes, where it takes an attribute
        as its only parameter and returns another column object.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The query with function applied.
    """
    org_col_names = list(t.c)
    cols = get_attributes(t, cols)
    col_names = [c.name for c in cols]

    # Apply function to columns
    name_d = dict()
    trimmed = []
    for c in t.c:
        if c not in cols:
            continue
        new_name = "temp" + c.name + "temp"
        name_d[new_name] = c.name
        trimmed.append(fn(c).label("temp" + c.name + "temp"))

    # Append new columns
    subquery = select(t, *trimmed).subquery()

    # Drop old columns
    subquery = drop_attributes(subquery, col_names)

    # Rename those temporary columns
    subquery = rename_attributes(subquery, name_d).subquery()

    # Re-order the columns as they were originally
    query = reorder_attributes(subquery, org_col_names, assert_same=False)

    return query


def trim_attributes(
    t: Union[Select, Subquery, Table, DBTable], cols: List[Union[str, Column]]
):
    """Returns query with columns listed having their
    leading/trailing whitespace trimmed (stripped).

    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or DBTable
        The query.
    cols : list of str or list of sqlalchemy.sql.schema.Column
        Attributes to trim.

    Returns
    -------
    sqlalchemy.sql.selectable.Select
        The query with trimmed attrbutes.
    """
    return apply_to_attributes(
        t, cols, lambda x: process_attribute(x, to_str=True, trim=True)
    )


def rga(o, *attr_args):
    """Recursive getattr (rga) is designed to express a
    series of attribute accesses with strings.

    E.g., o.a.b.c == rga(o, "a", "b", "c")

    Parameters
    ----------
    o: any
        Inital object.
    *attr_args : list of str
        Ordered list of attributes to access.

    Returns
    -------
    any
        The object accessed by the final attribute.
    """
    # Get attribute
    try:
        next_attr = getattr(o, attr_args[0])
    except ValueError:
        raise ValueError("No such attribute {}".format(attr_args[0]))

    # Base case
    if len(attr_args) == 1:
        return next_attr

    # Recurse
    return getattr(next_attr, attr_args[1:])


# === CONDITIONS ===
def process_elem(e: Any, **kwargs: dict) -> Any:
    """Preprocess some basic object such as an integer, float, or string.

    Parameters
    ----------
    e : any
        An element such as an integer, float, or string.
    **kwargs : dict, optional
        Preprocessing keyword arguments.

    Returns
    -------
    Any
        The preprocessed element.
    """
    # Extract kwargs
    lower = kwargs.get("lower", False)
    trim = kwargs.get("trim", False)
    to_str = kwargs.get("to_str", False)
    to_int = kwargs.get("to_int", False)
    to_float = kwargs.get("to_float", False)

    # Convert to string
    if to_str:
        e = str(e)

    # If a string
    if isinstance(e, str):
        if lower:
            e = e.lower()

        if trim:
            e = e.strip()

    # Convert to int
    if to_int:
        e = int(e)

    # Convert to float
    if to_float:
        e = float(e)

    return e


def process_list(lst: Union[Any, List[Any]], **kwargs: dict) -> List[Any]:
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
    # Convert potentially non-list variable to list
    lst = to_list(lst)

    # Process elements
    return [process_elem(i, **kwargs) for i in lst]


def process_attribute(col: Column, **kwargs: dict) -> Column:
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
    # Extract kwargs
    lower = kwargs.get("lower", False)
    trim = kwargs.get("trim", False)
    to_str = kwargs.get("to_str", False)
    to_int = kwargs.get("to_int", False)
    to_float = kwargs.get("to_float", False)

    # Convert to string
    if to_str:
        col = cast(col, String)

    # If a string column
    if "VARCHAR" in str(col.type):
        # Lower column
        if lower:
            col = func.lower(col)

        # Trim whitespace
        if trim:
            col = func.trim(col)

    if to_int:
        col = cast(col, Integer)

    if to_float:
        col = cast(col, Float)

    return col


def equals_cond(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: dict
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


def string_format_cond(
    col: Column, value: Any, fmt: str, to_str: bool = True, **kwargs: dict
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
    return process_attribute(col, to_str=True, **kwargs).like(
        fmt.format(process_elem(value, to_str=True, **kwargs))
    )


def substring_cond(
    col: Column, substring: Any, lower: bool = True, **kwargs: dict
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
    return string_format_cond(col, substring, "%%{}%%", lower=lower, **kwargs)


def startswith_cond(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: dict
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
    return string_format_cond(col, value, "{}%%", lower=lower, trim=trim, **kwargs)


def endswith_cond(
    col: Column, value: Any, lower: bool = True, trim: bool = True, **kwargs: dict
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
    return string_format_cond(col, value, "%%{}", lower=lower, trim=trim, **kwargs)


def in_list_condition(
    col: Column, lst: List[Any], lower: bool = True, trim: bool = True, **kwargs: dict
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
