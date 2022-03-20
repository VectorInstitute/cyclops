import numpy as np
import sqlalchemy
from sqlalchemy import cast
from sqlalchemy import String, Integer, Float
from sqlalchemy import select, func
from sqlalchemy.sql.selectable import Select, Subquery
from sqlalchemy.sql.schema import Table, Column

from dataclasses import dataclass

import logging
from cyclops.utils.log import setup_logging
from codebase_ops import get_log_file_path

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)

from functools import wraps
from typing import List, Any, Callable, Union

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


def _to_subquery(t: Union[Select, Subquery, Table, DBTable]) -> Subquery:
    """
    Converts a query from some type in QUERY_TYPES to
    type sqlalchemy.sql.selectable.Subquery.
    
    Parameters
    ----------
    t: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table
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

    raise ValueError("""t has type {}, but must have one of the
        following types:""".format(type(t), ', '.join(QUERY_TYPES)))


def _to_select(t: Union[Select, Subquery, Table, DBTable]) -> Select:
    """
    Converts a query from some type in QUERY_TYPES to
    type sqlalchemy.sql.selectable.Select.
    
    Parameters
    ----------
    t: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table
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
    
    raise ValueError("""t has type {}, but must have one of the
        following types:""".format(type(t), ', '.join(QUERY_TYPES)))


# Dictionary mapping query type -> query type conversion function
QUERY_TO_TYPE_FNS = {
    Subquery: _to_subquery,
    Select: _to_select,
    Table: lambda x: x,
    DBTable: lambda x: x,
}


def param_types_to_type(relevant_types: List[Any], to_type_fn: Callable) -> Callable:
    """
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
    to_type : Any
        The type to which to convert.

    Returns
    -------
    Callable
        The processed function.
    """
    if not to_type in QUERY_TYPES:
        raise ValueError("to_type must be in {}".format(QUERY_TYPES))
    
    to_type_fn = QUERY_TO_TYPE_FNS[to_type]
    return param_types_to_type(QUERY_TYPES, to_type_fn)



# === COLUMN PREPROCESSING ===
def to_list(a):
    # Convert non-list variable to list
    if isinstance(a, np.ndarray):
        a = list(a)
    elif not isinstance(a, list):
        a = [a]
    return a


@query_params_to_type(Subquery)
def get_attribute(t: Union[Select, Subquery, Table, DBTable], \
    a: Union[str, Column], assert_same: bool = True):
    """Extracts an attribute from the subquery.
    
    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
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
                raise ValueError("""Column provided and that of the same name
                    in the query do not match. This assertion can be ignored by
                    setting assert_same=False.""")
        return a

    # Otherwise, a string
    if not a in col_names:
        raise ValueError("Query does not contain column {}".format(a))
    
    return t.c[col_names.index(a)]


@query_params_to_type(Subquery)
def get_attributes(t: Union[Select, Subquery, Table, DBTable], \
    a: Union[str, Column, List[str], List[Column]], \
    assert_same: bool = True):
    """Extracts an attribute from the subquery.
    
    Parameters
    ----------
    t : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery or sqlalchemy.sql.schema.Table or DBTable
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
    return [get_attribute(t, aa, assert_same=assert_same) \
        for aa in to_list(a)]


def process_elem(s, **kwargs):
    # Extract kwargs
    lower = kwargs.get('lower', False)
    trim = kwargs.get('trim', False)
    to_str = kwargs.get('to_str', False)
    to_int = kwargs.get('to_int', False)
    to_float = kwargs.get('to_float', False)
    
    # Convert to string
    if to_str:
        s = str(s)
    
    # If a string
    if isinstance(s, str):
        if lower:
            s = s.lower()

        if trim:
            s = s.strip()
    
    # Convert to int
    if to_int:
        s = int(s)
    
    # Convert to float
    if to_float:
        s = float(s)
    
    return s


def process_list(a, **kwargs):
    # Convert non-list variable to list
    a = to_list(a)
    
    # Process elements
    return [process_elem(i, **kwargs) for i in a]


def process_col(col, **kwargs):
    # Extract kwargs
    lower = kwargs.get('lower', False)
    trim = kwargs.get('trim', False)
    to_str = kwargs.get('to_str', False)
    to_int = kwargs.get('to_int', False)
    to_float = kwargs.get('to_float', False)
    
    # Convert to string
    if to_str:
        col = cast(col, String)
    
    # If a string column
    if 'VARCHAR' in str(col.type):
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


# === CONDITIONS ===
def equals_cond(col, value, lower=True, trim=True, **kwargs):
    """
    Assumes that, if searching for a string, both the value and column values
    should be converted to lowercase and trimmed of leading/trailing whitespace.
    """
    return process_col(col, lower=lower, trim=trim, **kwargs) == \
        process_elem(value, lower=lower, trim=trim, **kwargs)
    
    
def substring_cond(col, substring, to_str=True, lower=True, **kwargs):
    return process_col(col, to_str=to_str, lower=lower, **kwargs).like( \
        '%%{}%%'.format(process_elem(substring, to_str=to_str, lower=lower, **kwargs)))


def startswith_cond(col, value, to_str=True, lower=True, trim=True, **kwargs):
    return process_col(col, to_str=to_str, lower=lower, trim=trim, **kwargs).like( \
        '{}%%'.format(process_elem(value, to_str=to_str, lower=lower, trim=trim, **kwargs)))


def endswith_cond(col, value, to_str=True, lower=True, trim=True, **kwargs):
    return process_col(col, to_str=to_str, lower=lower, trim=trim, **kwargs).like( \
        '%%{}'.format(process_elem(value, to_str=to_str, lower=lower, trim=trim, **kwargs)))


def in_list_condition(col, lst, lower=True, trim=True, **kwargs):
    """
    Assumes that, if searching for a string, both the value and column values
    should be converted to lowercase and trimmed of leading/trailing whitespace.
    """
    return process_col(col, lower=lower, trim=trim, **kwargs).in_( \
        process_list(lst, lower=lower, trim=trim, **kwargs))


# === SELECT ===
@query_params_to_type(Subquery)
def drop_attributes(t, drop_cols):   
    drop_cols = get_attributes(t, drop_cols)
    return select(*[c for c in t.c if not c in drop_cols])


@query_params_to_type(Subquery)
def rename_attributes(t, d):
    return select(*[c.label(d[c.name]) if c.name in d else c for c in t.c])


@query_params_to_type(Subquery)
def reorder_attributes(t, cols, assert_same=True):
    old_order = [c.name for c in t.c]
    new_order = [c.name for c in \
        get_attributes(t, cols, assert_same=assert_same)]
    
    if not set(old_order) == set(new_order):
        raise ValueError( \
            "Must specify all the query's attributes ({}) to re-order, not {}."
            .format(', '.join(old_order), ', '.join(new_order)))
    
    new_cols = []
    for c in new_order:
        new_cols.append(t.c[old_order.index(c)])
    
    return select(*new_cols)


@query_params_to_type(Subquery)
def apply_to_attributes(t, cols, fn):
    """
    Applies a function to some attributes.
    
    It is quite awkward to apply a function to attributes
    because you cannot simultaneously add/drop the columns,
    otherwise it treats the added columns as being joined
    through a Cartesian product.
    
    
    """
    org_col_names = list(t.c)
    cols = get_attributes(t, cols)
    col_names = [c.name for c in cols]

    # Apply function to columns
    name_d = dict()
    trimmed = []
    for c in t.c:
        if not c in cols:
            continue
        new_name = "temp" + c.name + "temp"
        name_d[new_name] = c.name
        trimmed.append(fn(c).label("temp" + c.name + "temp"))
    
    # Append trimmed columns
    subquery = select(t, *trimmed).subquery()

    # Drop old columns
    subquery = drop_attributes(subquery, col_names)
    
    # Rename those temporary columns
    subquery = rename_attributes(subquery, name_d).subquery()
    
    # Re-order the columns as they were originally
    query = reorder_attributes(subquery, org_col_names, assert_same=False)
    
    return query


def trim_columns(t, cols):
    return apply_to_attributes(
        t, cols, lambda x: process_col(x, trim=True)
    )


def rga(o, *attr_args):
    """
    Recursive getattr function.
    
    E.g., o.a.b.c == rga(t, "a", "b", "c")
    """
    next_attr = getattr(t, attr_args[0])
    if len(attr_args) == 1:
        return next_attr
    
    return getattr(next_attr, attr_args[1:])