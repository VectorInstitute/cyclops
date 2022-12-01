"""Low-level query processing functionality.

This module contains query operations functions such as AddColumn, Join, DropNulls, etc.
which can be used in high-level query API functions specific to datasets.

"""

# pylint: disable=too-many-lines

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from sqlalchemy import and_, cast, extract, func, or_, select
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
    has_columns,
    has_substring,
    in_,
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


@dataclass
class QAP:
    """Query argument placeholder (QAP) class.

    Parameters
    ----------
    kwarg_name: str
        Name of keyword argument for which this classs
        acts as a placeholder.
    required: bool
        Whether the keyword argument is required to run the process.
    transform_fn: callable
        A function accepting and transforming the passed in argument.

    """

    kwarg_name: str
    required: bool = True
    transform_fn: Optional[Callable] = None

    def __repr__(self):
        """Return the name of the placeholded keyword argument.

        Returns
        -------
        str
            The name of the keyword argument.

        """
        return self.kwarg_name

    def __call__(self, kwargs):
        """Recover the value of the placeholder argument.

        Attributes
        ----------
        kwargs: dict
            Dictionary containing self.kwarg_name as a key with a
            corresponding value.

        Returns
        -------
        any
            Value of the placeholder argument.

        """
        val = kwargs[self.kwarg_name]
        if self.transform_fn is not None:
            return self.transform_fn(val)

        return val


def ckwarg(kwargs, kwarg):
    """Get the value of a conditional keyword argument.

    A keyword argument may or may not be specified in some
    keyword arguments. If specified, return the value,
    otherwise return None.

    Parameters
    ----------
    kwargs: dict
        Process keyword arguments.
    kwarg: str
        The keyword argument of interest.

    Returns
    -------
    any, optional
        The value of the keyword argument if it exists, otherwise None.

    """
    if kwarg in kwargs:
        return kwargs[kwarg]
    return None


def remove_kwargs(process_kwargs, kwargs: Union[str, List[str]]):
    """Remove some keyword arguments from process_kwargs if they exist.

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


@table_params_to_type(Subquery)
def process_operations(  # pylint: disable=too-many-locals
    table: TableTypes, operations: List[tuple], user_kwargs: dict
) -> Subquery:
    """Query MIMIC encounter-specific patient data.

    Parameters
    ----------
    table: cyclops.query.util.TableTypes
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

    def get_required_qap(qaps):
        return [qap for qap in qaps if qap.required]

    def get_qap_args(operation, required=False):
        qap_args = [arg for arg in operation[1] if isinstance(arg, QAP)]
        if required:
            qap_args = get_required_qap(qap_args)
        return qap_args

    def get_qap_kwargs(operation, required=False):
        qap_kwargs = [val for _, val in operation[2].items() if isinstance(val, QAP)]
        if required:
            qap_kwargs = get_required_qap(qap_kwargs)
        return qap_kwargs

    def process_args(args):
        return [arg(user_kwargs) if isinstance(arg, QAP) else arg for arg in args]

    def process_kwargs(kwargs):
        for key, value in kwargs.items():
            if isinstance(value, QAP):
                # Convert if found
                if str(value) in user_kwargs:
                    kwargs[key] = value(user_kwargs)
                # Otherwise, ensure it wasn't required and remove later
                else:
                    if value.required:
                        raise ValueError(f"QAP {str(value)} must be specified.")

        # Remove optional kwargs which were not specified
        kwargs = {
            key: value for key, value in kwargs.items() if not isinstance(value, QAP)
        }

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
        required = get_qap_args(operation, required=True) + get_qap_kwargs(
            operation, required=True
        )
        specified_required = [str(r) in list(user_kwargs.keys()) for r in required]

        if not all(specified_required):
            # If not all the required parameters are specified ones, warn if only
            # some of the parameters are.
            all_params = get_qap_args(operation, required=False) + get_qap_kwargs(
                operation, required=False
            )
            specified_params = [str(r) in list(user_kwargs.keys()) for r in all_params]
            if any(specified_params):
                specified_kwargs = [
                    str(kwarg)
                    for i, kwarg in enumerate(all_params)
                    if specified_params[i]
                ]
                missing_kwargs = [
                    str(kwarg)
                    for i, kwarg in enumerate(required)
                    if not specified_required[i]
                ]
                raise ValueError(
                    f"""Process arguments {', '.join(specified_kwargs)} were
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
    return FilterColumns(keep_cols)(table)


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
    table: TableTypes,
    cols: Optional[Union[str, List[str]]] = None,
    cols_not_in: Optional[Union[str, List[str]]] = None,
    timestamp_cols: Optional[Union[str, List[str]]] = None,
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
class Drop:  # pylint: disable=too-few-public-methods
    """Drop some columns.

    Parameters
    ----------
    cols: str or list of str
        Columns to drop.

    """

    cols: Union[str, List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return drop_columns(table, self.cols)


@dataclass
class Rename:  # pylint: disable=too-few-public-methods
    """Rename some columns.

    Parameters
    ----------
    rename_map: dict
        Map from an existing column name to another name.
    check_exists: bool
        Whether to check if all of the keys in the map exist as columns.

    """

    rename_map: dict
    check_exists: bool = True

    def __call__(self, table: TableTypes) -> Subquery:
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
        if self.check_exists:
            table = process_checks(table, cols=list(self.rename_map.keys()))
        return rename_columns(table, self.rename_map)


@dataclass
class Substring:  # pylint: disable=too-few-public-methods
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
        table = select(
            table,
            func.substr(
                get_column(table, self.col), self.start_index, self.stop_index
            ).label(self.new_col_label),
        ).subquery()

        return table


@dataclass
class Reorder:  # pylint: disable=too-few-public-methods
    """Reorder the columns in a table.

    Parameters
    ----------
    cols: list of str
        Complete list of table column names in the new order.

    """

    cols: List[str]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return reorder_columns(table, self.cols)


class ReorderAfter:  # pylint: disable=too-few-public-methods
    """Reorder a number of columns to come after a specified column.

    cols: list of str
        Ordered list of column names which will come after a specified column.
    after: str
        Column name for the column after which the other columns will follow.

    """

    def __init__(self, cols: Union[str, List[str]], after: str):
        """Initialize."""
        self.cols = to_list(cols)
        self.after = after

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        self.cols = to_list(self.cols)
        table = process_checks(table, cols=self.cols + [self.after])
        names = get_column_names(table)
        names = [name for name in names if name not in self.cols]
        name_after_ind = names.index(self.after) + 1
        new_order = names[:name_after_ind] + self.cols + names[name_after_ind:]
        return Reorder(new_order)(table)


@dataclass
class FilterColumns:  # pylint: disable=too-few-public-methods
    """Keep only the specified columns in a table.

    Parameters
    ----------
    cols: str or list of str
        The columns to keep.

    """

    cols: Union[str, List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return filter_columns(table, self.cols)


@dataclass
class Trim:  # pylint: disable=too-few-public-methods
    """Trim the whitespace from some string columns.

    Parameters
    ----------
    cols: str or list of str
        Columns to trim.
    new_col_labels: str or list of str, optional
        If specified, create new columns with these labels. Otherwise,
        apply the function to the existing columns.

    """

    cols: Union[str, List[str]]
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)
        return trim_columns(table, self.cols, new_col_labels=self.new_col_labels)


@dataclass
class Literal:  # pylint: disable=too-few-public-methods
    """Add a literal column to a table.

    Parameters
    ----------
    value: any
        Value of the literal, e.g., a string or integer.
    col: str
        Label of the new literal column.

    """

    value: Any
    col: str

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols_not_in=self.col)
        return select(table, literal(self.value).label(self.col)).subquery()


@dataclass
class ExtractTimestampComponent:  # pylint: disable=too-few-public-methods
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

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
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
class AddNumeric:  # pylint: disable=too-few-public-methods
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

    add_to: Union[str, List[str]]
    num: Union[int, float]
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.add_to, cols_not_in=self.new_col_labels)
        return apply_to_columns(
            table,
            self.add_to,
            lambda x: x + self.num,
            new_col_labels=self.new_col_labels,
        )


@dataclass
class AddDeltaConstant:  # pylint: disable=too-few-public-methods
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

    add_to: Union[str, List[str]]
    delta: timedelta
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
            table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
        )

        return apply_to_columns(
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

    """

    add_to: Union[str, List[str]]
    col: str
    negative: Optional[bool] = False
    new_col_labels: Optional[Union[str, List[str]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
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
            table = process_checks(
                table, timestamp_cols=self.add_to, cols_not_in=self.new_col_labels
            )
        else:
            table = process_checks(
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

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
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
class Cast:
    """Cast columns to a specified type.

    Currently supporting conversions to str, int, float, date and timestamp.

    Parameters
    ----------
    cols : str or list of str
        Columns to = cast.
    type_ : str
        Name of type to which to convert. Must be supported.

    """

    cols: Union[str, List[str]]
    type_: str

    def __call__(self, table: TableTypes):
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.cols)

        cast_type_map = {
            "str": "to_str",
            "int": "to_int",
            "float": "to_float",
            "date": "to_date",
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


class Join:  # pylint:disable=too-few-public-methods, too-many-arguments
    """Join a table with another table.

    Warning: If neither on nor cond parameters are specified, an
    expensive Cartesian product is performed.

    Attributes
    ----------
    join_table: cyclops.query.util.TableTypes
        Table on which to join.
    on_: list of str or tuple, optional
        A list of strings or tuples representing columns on which to join.
        Strings represent columns of same name in both tables. A tuple of
        style (table_col, join_table_col) is used to join on columns of
        different names. Suggested to specify this parameter as opposed to
        cond.
    on_to_type: list of type, optional
        A list of types to which to convert the on_ columns before joining. Useful when
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

    @table_params_to_type(Subquery)
    def __init__(
        self,
        join_table: TableTypes,
        on: Union[  # pylint:disable=invalid-name
            str, List[str], tuple, List[tuple], None
        ] = None,
        on_to_type: Union[type, List[type], None] = None,
        cond: Union[BinaryExpression, None] = None,
        table_cols: Union[str, List[str], None] = None,
        join_table_cols: Union[str, List[str], None] = None,
        isouter: Optional[bool] = False,
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

        Paramaters
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
            table = process_checks(table, cols=none_add(self.table_cols, on_table_cols))
            self.join_table = process_checks(
                self.join_table, cols=none_add(self.join_table_cols, on_join_table_cols)
            )
            # Filter columns, keeping those being joined on
            table = append_if_missing(table, self.table_cols, on_table_cols)
            self.join_table = append_if_missing(
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
                table = FilterColumns(self.table_cols)(table)
            if self.join_table_cols is not None:
                self.join_table = FilterColumns(self.table_cols)(  # type: ignore
                    self.join_table
                )

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


class ConditionEquals:  # pylint: disable=too-few-public-methods
    """Filter rows based on being equal, or not equal, to some value.

    Attributes
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
        Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        value: Any,
        not_: bool = False,
        binarize_col: Optional[str] = None,
        **cond_kwargs,
    ):
        """Initialize."""
        self.col = col
        self.value = value
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = equals(get_column(table, self.col), self.value, **self.cond_kwargs)
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionIn:  # pylint: disable=too-few-public-methods
    """Filter rows based on having a value in list of values.

    Attributes
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
        Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        values: Union[Any, List[Any]],
        not_: bool = False,
        binarize_col: Optional[str] = None,
        **cond_kwargs,
    ):
        """Initialize."""
        self.col = col
        self.values = values
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = in_(
            get_column(table, self.col),
            to_list(self.values),
            **self.cond_kwargs,
        )
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionSubstring:  # pylint: disable=too-few-public-methods
    """Filter rows on based on having substrings.

    Can be specified whether it must have any or all of the specified substrings.
    This makes no difference when only one substring is provided

    Attributes
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
        Optional keyword arguments for processing the condition.

    """

    def __init__(
        self,
        col: str,
        substrings: Union[str, List[str]],
        any_: bool = True,
        not_: bool = False,
        binarize_col: Optional[str] = None,
        **cond_kwargs,
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

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        conds = [
            has_substring(get_column(table, self.col), sub, **self.cond_kwargs)
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


class ConditionStartsWith:  # pylint: disable=too-few-public-methods
    """Filter rows based on starting with some string.

    Attributes
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

    """

    def __init__(
        self,
        col: str,
        string: str,
        not_: bool = False,
        binarize_col: Optional[str] = None,
        **cond_kwargs,
    ):
        """Initialize."""
        self.col = col
        self.string = string
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = starts_with(get_column(table, self.col), self.string, **self.cond_kwargs)
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionEndsWith:  # pylint: disable=too-few-public-methods
    """Filter rows based on ending with some string.

    Attributes
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

    """

    def __init__(
        self,
        col: str,
        string: str,
        not_: bool = False,
        binarize_col: Optional[str] = None,
        **cond_kwargs,
    ):
        """Initialize."""
        self.col = col
        self.string = string
        self.not_ = not_
        self.binarize_col = binarize_col
        self.cond_kwargs = cond_kwargs

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(table, cols=self.col, cols_not_in=self.binarize_col)
        cond = ends_with(get_column(table, self.col), self.string, **self.cond_kwargs)
        if self.not_:
            cond = cond._negate()

        if self.binarize_col is not None:
            return select(
                table, cast(cond, Boolean).label(self.binarize_col)
            ).subquery()

        return select(table).where(cond).subquery()


class ConditionInYears:  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp column being in a list of years.

    Attributes
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
        years: Union[int, List[int]],
        not_: bool = False,
        binarize_col: Optional[str] = None,
    ):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.years = years
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
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


class ConditionInMonths:  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp being in a list of years.

    Attributes
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
        months: Union[int, List[int]],
        not_: bool = False,
        binarize_col: Optional[str] = None,
    ):
        """Initialize."""
        self.timestamp_col = timestamp_col
        self.months = months
        self.not_ = not_
        self.binarize_col = binarize_col

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        table = process_checks(
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


class ConditionBeforeDate:  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp being before some date.

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

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
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
            .where(get_column(table, self.timestamp_col) <= timestamp)
            .subquery()
        )


class ConditionAfterDate:  # pylint: disable=too-few-public-methods
    """Filter rows based on a timestamp being after some date.

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

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
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
            .where(get_column(table, self.timestamp_col) >= timestamp)
            .subquery()
        )


@dataclass
class Limit:  # pylint: disable=too-few-public-methods
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

        Paramaters
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
class RandomizeOrder:
    """Randomize order of table rows.

    Useful when the data is ordered, so certain rows cannot
    be seen or analyzed when limited.

    Warning: Becomes quite slow on large tables.

    """

    @table_params_to_type(Subquery)
    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
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
class DropNulls:
    """Remove rows with null values in some specified columns.

    Parameters
    ----------
    cols: str or list of str
        Columns in which, if a value is null, the corresponding row
        is removed.

    """

    cols: Union[str, List[str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
        ----------
        table : cyclops.query.util.TableTypes
            Table on which to perform the operation.

        Returns
        -------
        sqlalchemy.sql.selectable.Subquery
            Processed table.

        """
        self.cols = to_list(self.cols)
        table = process_checks(table, cols=self.cols)

        cond = and_(*[not_equals(get_column(table, col), None) for col in self.cols])
        return select(table).where(cond).subquery()


@dataclass
class OrderBy:
    """Order, or sort, the rows of a table by some columns.

    Attributes
    ----------
    cols: str or list of str
        Columns by which to order.
    ascending: bool or list of bool
        Whether to order each columns by ascending (True) or descending (False).
        If not provided, orders all by ascending.

    """

    cols: Union[str, List[str]]
    ascending: Optional[Union[bool, List[bool]]] = None

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
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
        table = process_checks(table, cols=self.cols)

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
class GroupByAggregate:
    """Aggregate over a group by object.

    Attributes
    ----------
    groupby_cols: str or list of str
        Columns by which to group.
    aggfuncs: dict
        Specify a dictionary of key-value pairs:
        column name: aggfunc string or
        column name: (aggfunc string, new column label)
        This labelling allows for the aggregation of the same column with
        different functions.

    """

    groupby_cols: Union[str, List[str]]
    aggfuncs: Union[Dict[str, Sequence[str]], Dict[str, str]]

    def __call__(self, table: TableTypes) -> Subquery:
        """Process the table.

        Paramaters
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
        table = process_checks(table, cols=groupby_names + aggfunc_cols)

        # Error checking
        for aggfunc_str in aggfunc_strs:
            if aggfunc_str not in str_to_aggfunc:
                allowed_strs = ", ".join(list(str_to_aggfunc.keys()))
                raise ValueError(
                    f"Invalid aggfuncs specified. Allowed values are {allowed_strs}."
                )

        all_names = groupby_names + aggfunc_names
        if len(all_names) != len(set(all_names)):
            raise ValueError(
                """Duplicate column names were found. Try naming aggregated columns
                to avoid this issue."""
            )

        # Perform group by
        groupby_cols = get_columns(table, groupby_names)
        agg_cols = get_columns(table, aggfunc_cols)
        agg_cols = [
            str_to_aggfunc[aggfunc_strs[i]](col).label(aggfunc_names[i])
            for i, col in enumerate(agg_cols)
        ]

        return select(*groupby_cols, *agg_cols).group_by(*groupby_cols).subquery()
