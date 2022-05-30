"""High-level query processing functionality."""

# pylint: disable=too-many-lines

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List, Optional, Union

from sqlalchemy import and_, extract, select
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql.selectable import Subquery

# Logging.
from codebase_ops import get_log_file_path
from cyclops.query.util import (
    QueryTypes,
    apply_to_attributes,
    check_timestamp_attributes,
    drop_attributes,
    equals,
    filter_attributes,
    get_attribute,
    get_attributes,
    get_delta_attribute,
    has_attributes,
    has_substring,
    in_,
    not_equals,
    query_params_to_type,
    rename_attributes,
    reorder_attributes,
    to_datetime_format,
    to_list,
    to_list_optional,
    trim_attributes,
)
from cyclops.utils.log import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class QAP:
    """Query argument placeholder (QAP) class.

    Attributes
    ----------
    kwarg_name: str
        Name of keyword argument for which this classs
        acts as a placeholder.

    """

    kwarg_name: str

    def __repr__(self):
        """Return the name of the placeholded keyword argument.

        Returns
        -------
        str
            The name of the keyword argument.

        """
        return self.kwarg_name


def ckwarg(process_kwargs, kwarg):
    """Get the value of a conditional keyword argument.

    A keyword argument may or may not be specified in some
    keyword arguments. If specified, return the value,
    otherwise return None.

    Parameters
    ----------
    process_kwargs: dict
        Process keyword arguments.
    kwarg: str
        The keyword argument of interest.

    Returns
    -------
    any, optional
        The value of the keyword argument if it exists, otherwise None.

    """
    if kwarg in process_kwargs:
        return process_kwargs[kwarg]
    return None


def remove_kwargs(process_kwargs, kwargs: Union[str, List[str]]):
    """Remove some keyword arguments from process_kwargs.

    Parameters
    ----------
    process_kwargs: dict
        Process keyword arguments.
    kwargs: str or list of str
        The keyword arguments to remove should they exist.

    """
    kwargs = to_list(kwargs)
    for kwarg in kwargs:
        if kwarg in process_kwargs:
            del process_kwargs[kwarg]
    return process_kwargs


@query_params_to_type(Subquery)
def process_operations(  # pylint: disable=too-many-locals
    table: QueryTypes, operations: List[tuple], user_kwargs: dict
) -> Subquery:
    """Query MIMIC encounter-specific patient data.

    Parameters
    ----------
    table: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        Table to process.
    operations:
        Operations to execute, which are of the form:
        (process function, arguments to pass, keyword arguments to pass).
    user_kwargs:
        Keyword arguments specified by the calling function, or user.
        If a keyword argument is None, it is discarded.

    Returns
    -------
    cyclops.query.interface.QueryInterface
        Constructed query, wrapped in an interface object.

    """
    # Remove None user kwargs
    user_kwargs = {
        kwarg: value for kwarg, value in user_kwargs.items() if value is not None
    }

    def get_qap_args(operation):
        return [arg for arg in operation[1] if isinstance(arg, QAP)]

    def get_qap_kwargs(operation):
        return [kwarg for kwarg in operation[2] if isinstance(kwarg, QAP)]

    def process_args(args):
        return [user_kwargs[str(arg)] if isinstance(arg, QAP) else arg for arg in args]

    def process_kwargs(kwargs):
        for key, value in kwargs.items():
            if isinstance(value, QAP):
                kwargs[key] = user_kwargs[str(value)]
        return kwargs

    def flatten_2d(lst):
        return [j for sub in lst for j in sub]

    # Get the valid kwargs which may be specified for this function
    qap_args = flatten_2d([get_qap_args(op) for op in operations])
    qap_kwargs = flatten_2d([get_qap_kwargs(op) for op in operations])
    kwargs_supported = [str(qap) for qap in qap_args + qap_kwargs]

    # Ensure only supported operations are being performed
    for kwarg in user_kwargs:
        if kwarg not in kwargs_supported:
            raise ValueError(
                f"""Keyword {kwarg} is not supported in this query function,
                only keywords {', '.join(kwargs_supported)}"""
            )

    for operation in operations:
        process_func, args, kwargs = operation

        # Skip if not all the required kwargs were given
        required = get_qap_args(operation)
        specified = [str(r) in list(user_kwargs.keys()) for r in required]
        if not all(specified):
            # Warn if some of required are specified, but not all
            if any(specified):
                # STILL NEED TO TEST
                specified_kwargs = [
                    kwarg for i, kwarg in enumerate(required) if specified[i]
                ]
                missing_kwargs = [
                    kwarg for i, kwarg in enumerate(required) if specified[i]
                ]
                raise ValueError(
                    f"""Process arguments {', '.join(specified_kwargs)} were partially
                    specified, but missing {', '.join(missing_kwargs)}"""
                )
            continue

        # Run the operation, replacing all the query argument placeholders with
        # their respective values
        args = process_args(args)
        kwargs = process_kwargs(kwargs)
        table = process_func(*args, **kwargs)(table)

    return table


def col_name_remove(lst, remove_names):
    """Remove any Column from lst if its name is in remove_names."""
    return [x for x in lst if x.name not in remove_names]


def append_if_missing(table, keep_cols, force_include_cols):
    """Keep only certain columns in a table, but must include certain columns.

    Parameters
    ----------
    table:
    keep_cols
    force_include_cols

    """
    if keep_cols is None:
        return table

    extend_cols = [col for col in force_include_cols if col not in keep_cols]
    keep_cols = extend_cols + keep_cols
    return select(*get_attributes(table, keep_cols)).subquery()


def none_add(obj1, obj2):
    """Add two objects together while ignoring None values.

    If both objects are None, returns None.

    Parameters
    ----------
    obj1: Any
    obj2: Any

    """
    if obj1 is None:
        return obj2
    if obj2 is None:
        return obj1
    return obj1 + obj2


def process_checks(
    table: QueryTypes,
    cols: Optional[Union[str, List[str]]] = None,
    cols_not_in: Optional[Union[str, List[str]]] = None,
    timestamp_cols: Optional[Union[str, List[str]]] = None,
) -> Subquery:
    """Perform checks, and possibly alterations, on a table.

    Paramaters
    ----------
    table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
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
        has_attributes(table, cols, raise_error=True)

    if cols_not_in is not None:
        cols_not_in = to_list(cols_not_in)
        if has_attributes(table, cols_not_in, raise_error=False):
            raise ValueError(f"Table cannot have columns {cols_not_in}")

    if timestamp_cols is not None:
        timestamp_cols = to_list(timestamp_cols)
        has_attributes(table, timestamp_cols, raise_error=True)
        check_timestamp_attributes(table, timestamp_cols, raise_error=True)

    return table


@dataclass
class Drop:  # pylint: disable=too-few-public-methods
    """Drop some columns.

    Attributes
    ----------
    cols: str or list of str
        Columns to drop.

    """

    cols: Union[str, List[str]]

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return drop_attributes(table, self.cols)


@dataclass
class Rename:  # pylint: disable=too-few-public-methods
    """Rename some columns.

    Attributes
    ----------
    rename_map: dict
        Map from an existing column name to another name.
    check_keys: bool
        Whether to check if all of the keys in the map exist as columns.

    """

    rename_map: dict
    check_keys: bool = True

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        if self.check_keys:
            table = process_checks(table, cols=list(self.rename_map.keys()))
        return rename_attributes(table, self.rename_map)


@dataclass
class Reorder:  # pylint: disable=too-few-public-methods
    """Reorder the columns in a table.

    Attributes
    ----------
    cols: list of str
        Complete list of table column names in the new order.

    """

    cols: List[str]

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return reorder_attributes(table, self.cols)


@dataclass
class FilterColumns:  # pylint: disable=too-few-public-methods
    """Keep only the specified columns in a table.

    Attributes
    ----------
    cols: str or list of str
        The columns to keep.

    """

    cols: Union[str, List[str]]

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return filter_attributes(table, self.cols)


# class ApplyLambda():
#    def __init__(self, func: Callable, ):


@dataclass
class Trim:  # pylint: disable=too-few-public-methods
    """Trim the whitespace from some string columns.

    Attributes
    ----------
    cols: str or list of str
        Columns to trim.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    cols: Union[str, List[str]]
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return trim_attributes(table, self.cols, new_col_labels=self.new_col_labels)


@dataclass
class AddNumeric:  # pylint: disable=too-few-public-methods
    """Add a numeric value to some columns.

    Attributes
    ----------
    add_to: str or list of str
        Column names specifying to which columns is being added.
    num: int or float
        Adds this value to the add_to columns.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    add_to: Union[str, List[str]]
    num: Union[int, float]
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.add_to, cols_not_in=self.new_col_labels)
        return apply_to_attributes(
            table,
            self.add_to,
            lambda x: x + self.num,
            new_col_labels=self.new_col_labels,
        )


@dataclass
class AddDeltaConstant:  # pylint: disable=too-few-public-methods
    """Construct and add a datetime.timedelta object to some columns.

    Attributes
    ----------
    add_to: str or list of str
        Column names specifying to which columns is being added.
    delta: datetime.timedelta
        A timedelta object.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    add_to: Union[str, List[str]]
    delta: timedelta
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
            table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
        )

        return apply_to_attributes(
            table,
            self.add_to,
            lambda x: x + self.delta,
            new_col_labels=self.new_col_labels,
        )


@dataclass
class AddColumn:  # pylint: disable=too-few-public-methods
    """Add an column to some columns.

    Pay attention to column types. Some combinations will work,
    whereas others will not.

    Attributes
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

    """

    add_to: Union[str, List[str]]
    col: str
    negative: Optional[bool] = False
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        # If the column being added is a timestamp column, ensure the others are too
        if check_timestamp_attributes(table, self.col):
            table = process_checks(
                table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
            )
        else:
            table = process_checks(
                table, cols=self.add_to, cols_not_in=self.new_col_labels
            )

        col = get_attribute(table, self.col)

        if self.negative:
            col = -col

        return apply_to_attributes(
            table,
            self.add_to,
            lambda x: x + col,
            new_col_labels=self.new_col_labels,
        )


class AddDeltaColumns:  # pylint: disable=too-few-public-methods
    """Construct and add an interval column to some columns.

    Attributes
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
        add_to: Union[str, List[str]],
        negative: Optional[bool] = False,
        new_col_labels: Optional[Union[str, List[str]]] = None,
        **delta_kwargs,
    ):
        """Initialize."""
        self.add_to = add_to
        self.negative = negative
        self.new_col_labels = new_col_labels
        self.delta_kwargs = delta_kwargs

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
            table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
        )

        delta = get_delta_attribute(table, **self.delta_kwargs)

        if self.negative:
            delta = -delta

        return apply_to_attributes(
            table,
            self.add_to,
            lambda x: x + delta,
            new_col_labels=self.new_col_labels,
        )


class Join:  # pylint:disable=too-few-public-methods, too-many-arguments
    """Join a table with another table.

    Warning: If neither on nor cond parameters are specified, an
    expensive Cartesian product is performed.

    Attributes
    ----------
    join_table: sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
    or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
        Table on which to join.
    on_: str or list of str, optional
        Columns of same name in both tables on which to join. It is
        highly suggested to specify this parameter as opposed to cond.
    cond: BinaryExpression, optional
        Condition on which to join to tables.
    table_attrs: str or list of str, optional
        Filters to keep only these columns from the table.
    join_table_attrs:
        Filters to keep only these columns from the join_table.

    """

    @query_params_to_type(Subquery)
    def __init__(
        self,
        join_table: QueryTypes,
        on: Optional[Union[str, List[str]]] = None,  # pylint:disable=invalid-name
        cond: Optional[BinaryExpression] = None,
        table_cols: Optional[Union[str, List[str]]] = None,
        join_table_cols: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize."""
        if on is not None and cond is not None:
            raise ValueError("Cannot specify both the 'on' and 'cond' arguments.")

        self.join_table = join_table
        self.cond = cond
        self.on_ = to_list_optional(on)
        self.table_cols = to_list_optional(table_cols)
        self.join_table_cols = to_list_optional(join_table_cols)

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=none_add(self.table_cols, self.on_))
        self.join_table = process_checks(
            self.join_table, cols=none_add(self.join_table_cols, self.on_)
        )

        # Filter columns
        table = append_if_missing(table, self.table_cols, self.on_)
        self.join_table = append_if_missing(
            self.join_table, self.join_table_cols, self.on_
        )

        # Join on the equality of values in columns of same name in both tables
        if self.on_ is not None:
            cond = and_(
                *[
                    get_attribute(table, col) == get_attribute(self.join_table, col)
                    for col in self.on_
                ]
            )
            table = select(table.join(self.join_table, cond))

        # Join on a specified condition
        elif self.cond is not None:
            table = select(table.join(self.join_table, self.cond))

        # Join on no condition
        else:
            LOGGER.warning("A cartesian product has occurred.")
            table = select(table, self.join_table)

        # Filter to include no duplicates
        return select(
            *[col for col in table.columns if "%(" not in col.name]
        ).subquery()


class ConditionEquals:  # pylint: disable=too-few-public-methods
    """Filter rows on column being equal to some value.

    Attributes
    ----------
    col: str
        Column name on which to condition.
    value: any
        Value to equal.
    **cond_kwargs: bool
        Optional keyword arguments for processing the condition.

    """

    def __init__(self, col: str, value: Any, **cond_kwargs):
        """Initialize."""
        self.col = col
        self.value = value
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col)
        return (
            select(table)
            .where(
                equals(get_attribute(table, self.col), self.value, **self.cond_kwargs)
            )
            .subquery()
        )


@dataclass
class ConditionNotEquals:  # pylint: disable=too-few-public-methods
    """Filter rows on column being not equal to some value.

    Attributes
    ----------
    col: str
        Column name on which to condition.
    value: any
        Value to not equal.
    **cond_kwargs: bool
        Optional keyword arguments for processing the condition.

    """

    def __init__(self, col: str, value: Any, **cond_kwargs):
        """Initialize."""
        self.col = col
        self.value = value
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col)
        return (
            select(table)
            .where(
                not_equals(
                    get_attribute(table, self.col), self.value, **self.cond_kwargs
                )
            )
            .subquery()
        )


@dataclass
class ConditionIn:  # pylint: disable=too-few-public-methods
    """Filter rows on column having a value in list of values.

    Attributes
    ----------
    col: str
        Column name on which to condition.
    values: any or list of any
        Values in which the column value must be.
    **cond_kwargs: bool
        Optional keyword arguments for processing the condition.

    """

    def __init__(self, col: str, values: Union[Any, List[Any]], **cond_kwargs):
        """Initialize."""
        self.col = col
        self.values = values
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col)
        return (
            select(table)
            .where(
                in_(
                    get_attribute(table, self.col),
                    to_list(self.values),
                    **self.cond_kwargs,
                )
            )
            .subquery()
        )


@dataclass
class ConditionSubstring:  # pylint: disable=too-few-public-methods
    """Filter rows on column having a substring.

    Attributes
    ----------
    col: str
        Column name on which to condition.
    substring: any
        Substring.
    **cond_kwargs: bool
        Optional keyword arguments for processing the condition.

    """

    def __init__(self, col: str, substring: str, **cond_kwargs):
        """Initialize."""
        self.col = col
        self.substring = substring
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col)
        return (
            select(table)
            .where(
                has_substring(
                    get_attribute(table, self.col), self.substring, **self.cond_kwargs
                )
            )
            .subquery()
        )


@dataclass
class ConditionInYears:  # pylint: disable=too-few-public-methods
    """Filter rows on a timestamp column being in a list of years.

    Attributes
    ----------
    timestamp_col: str
        Timestamp column name.
    years: int or list of int
        Years in which the timestamps must be.

    """

    def __init__(self, timestamp_col: str, years: Union[int, List[int]]):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.years = years

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.timestamp_col)
        return (
            select(table)
            .filter(
                in_(
                    extract("year", get_attribute(table, self.timestamp_col)),
                    to_list(self.years),
                )
            )
            .subquery()
        )


@dataclass
class ConditionInMonths:  # pylint: disable=too-few-public-methods
    """Filter rows on a timestamp column being in a list of years.

    Attributes
    ----------
    timestamp_col: str
        Timestamp column name.
    months: int or list of int
        Months in which the timestamps must be.

    """

    def __init__(self, timestamp_col: str, months: Union[int, List[int]]):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.months = months

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.timestamp_col)
        return (
            select(table)
            .filter(
                in_(
                    extract("month", get_attribute(table, self.timestamp_col)),
                    to_list(self.months),
                )
            )
            .subquery()
        )


@dataclass
class ConditionBeforeDate:  # pylint: disable=too-few-public-methods
    """Filter rows in a timestamp column before some date.

    Attributes
    ----------
    timestamp_col: str
        Timestamp column name.
    timestamp: str or datetime.datetime
        A datetime object or str in YYYY-MM-DD format.

    """

    def __init__(self, timestamp_col: str, timestamp: Union[str, datetime]):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.timestamp = timestamp

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, timestamp_cols=self.timestamp_col)

        if isinstance(self.timestamp, str):
            timestamp = to_datetime_format(self.timestamp)
        else:
            timestamp = self.timestamp

        return (
            select(table)
            .where(get_attribute(table, self.timestamp_col) <= timestamp)
            .subquery()
        )


@dataclass
class ConditionAfterDate:  # pylint: disable=too-few-public-methods
    """Filter rows in a timestamp column after some date.

    Attributes
    ----------
    timestamp_col: str
        Timestamp column name.
    timestamp: str or datetime.datetime
        A datetime object or str in YYYY-MM-DD format.

    """

    def __init__(self, timestamp_col: str, timestamp: Union[str, datetime]):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.timestamp = timestamp

    def __call__(self, table: QueryTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : sqlalchemy.sql.selectable.Select or sqlalchemy.sql.selectable.Subquery
        or sqlalchemy.sql.schema.Table or cyclops.query.utils.DBTable
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, timestamp_cols=self.timestamp_col)

        if isinstance(self.timestamp, str):
            timestamp = to_datetime_format(self.timestamp)
        else:
            timestamp = self.timestamp

        return (
            select(table)
            .where(get_attribute(table, self.timestamp_col) >= timestamp)
            .subquery()
        )
