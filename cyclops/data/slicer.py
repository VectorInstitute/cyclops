"""Functions and classes for creating subsets of Hugging Face datasets."""

import copy
import datetime
import itertools
import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from datasets.formatting.formatting import LazyBatch
from dateutil.parser import parse
from pyarrow import ArrowInvalid


@dataclass
class SliceSpec:
    """Specifications for creating a slices of a dataset.

    Parameters
    ----------
    spec_list : List[Union[Dict[str, Any], List[Dict[str, Any]]]], default=[{}]
        A list of slice specifications. Each specification is a dictionary mapping
        a column name to a slice specification for that column. A slice specification
        is a dictionary containing one or more of the following keys:

        - `value`: The exact value of a column to select. This can be a single value
          a list of values. If a list of values is provided, the slice selects rows
          where the column value is in the list. Time strings are supported (e.g.
          `"2021-01-01 00:00:00"`).
        - `min_value`: The minimum value of a column to select. Specifying the
          `min_inclusive` key indicates whether to include the minimum value in the
          range. This works for numerical and datetime columns. Time strings are
          supported. The default value is set to -inf, if `max_value` is specified and
          `min_value` is not.
        - `min_inclusive`: Boolean value to indicated whether to include the `min_value`
          in the range. If True, the slice selects rows where the value is greater than
          or equal to the `min_value`. Defaults to True.
        - `max_value`: The maximum value of a column to select. This works for numerical
          and datetime columns. Specifying the `max_inclusive` key indicates whether to
          include the maximum value in the range.  Time strings are supported.
          The default value is set to `inf`, if `min_value` is specified and `max_value`
          is not.
        - `max_inclusive`: Boolean value to indicated whether to include the `max_value`
          in the range. If True, the slice selects rows where the value is less than or
          equal to the `max_value`. Defaults to True.
        - `year`: A single (numerical or string) value or list of values for selecting
          rows in a datetime column where the year matches the value(s). Defaults to
          None.
        - `month`: A single (numerical) value or list of values between 1 and 12 for
          selecting rows in a datetime column where the month matches the value(s).
          Defaults to None.
        - `day`: A single (numerical) value or list of values between 1 and 31 for
          selecting rows in a datetime column where the day matches the value(s).
          Defaults to None.
        - `hour`: A single (numerical) value or list of values between 0 and 23 for
          selecting rows in a datetime column where the hour matches the value(s).
          Defaults to None.
        - `negate`: A boolean flag indicating whether to negate the slice. If True, the
          slice selects rows where the feature value does not match the specification.
          Defaults to False.
        - `keep_nulls`: A boolean flag indicating whether to keep rows where the value
          is null. If used in conjunction with `negate`, the slice selects rows where
          the value is not null. Can be used on its own. Defaults to False.
    intersections : List[Tuple[int]], int, optional, default=None
        An indication of slices to intersect. If a list of tuples is provided, the
        tuples should contain the indices of the slices to intersect. If an integer is
        provided, it will be passed as the argument `r` in `itertools.combinations`,
        and all combinations of `r` slices will be intersected. The intersections are
        created _before_ the slices are registered.
    include_overall : bool, default=True
        Whether to include an `overall` slice that selects all examples.
    validate : bool, default=True
        Whether to validate the column names in the slice specifications.
    column_names : List[str], optional, default=None
        List of column names in the dataset. If provided and `validate` is True, it is
        used to validate the column names in the slice specifications.


    Attributes
    ----------
    spec_list : List[Union[Dict[str, Any], List[Dict[str, Any]]]]
        List of slice specifications.
    include_overall : bool
        Whether to include an `overall` slice that selects all examples.
    validate : bool
        Whether to validate the column names in the slice specifications.
    column_names : List[str]
        List of column names in the dataset.
    _registry : Dict[str, Callable]
        Dictionary mapping slice names to functions that create the slice.

    Examples
    --------
    >>> from cyclops.data.slicer import SliceSpec
    >>> slice_spec = SliceSpec(
    ...     spec_list=[
    ...         {"feature_1": {"keep_nulls": False}},
    ...         {
    ...             "feature_2": {"keep_nulls": False},
    ...             "feature_3": {"keep_nulls": False},
    ...         },
    ...         {"feature_1": {"value": "value_1"}},
    ...         {"feature_1": {"value": ["value_1", "value_2"]}},
    ...         {"feature_1": {"value": "value_1", "negate": True, "keep_nulls": True}},
    ...         {"feature_1": {"min_value": "2020-01-01", "max_value": "2020-12-31"}},
    ...         {
    ...             "feature_1": {
    ...                 "min_value": 5,
    ...                 "max_value": 60,
    ...                 "min_inclusive": False,
    ...                 "max_inclusive": False,
    ...             }
    ...         },
    ...         {"feature_1": {"year": [2020, 2021, 2022]}},
    ...         {"feature_1": {"month": [6, 7, 8]}},
    ...         {"feature_1": {"month": 6, "day": 1}},
    ...         {"feature_1": {"contains": "value_1"}},
    ...         {"feature_1": {"contains": ["value_1", "value_2"]}},
    ...         {
    ...             "feature_1": {"value": "value_1"},
    ...             "feature_2": {
    ...                 "min_value": "2020-01-01",
    ...                 "keep_nulls": False,
    ...             },
    ...             "feature_3": {"year": ["2000", "2010", "2020"]},
    ...         },
    ...     ],
    ... )
    >>> for slice_name, slice_func in slice_spec.slices():
    ...     print(slice_name)
    ...     # do something with slice_func here (e.g. dataset.filter(slice_func))
    feature_1:non_null
    feature_2:non_null&feature_3:non_null
    feature_1:value_1
    feature_1:value_1, value_2
    !(feature_1:value_1)
    feature_1:[2020-01-01 - 2020-12-31]
    feature_1:(5 - 60)
    feature_1:year=[2020, 2021, 2022]
    feature_1:month=[6, 7, 8]
    feature_1:month=6, day=1
    feature_1:contains value_1
    feature_1:contains ['value_1', 'value_2']
    feature_1:value_1&feature_2:[2020-01-01 - inf]&feature_3:year=['2000', '2010', '2020']
    overall

    >>> # a different way to create intersections/compound slices
    >>> slice_spec = SliceSpec(
    ...     spec_list=[
    ...         {"feature_1": {"keep_nulls": False}},
    ...         {"feature_2": {"keep_nulls": False}},
    ...     ],
    ...     include_overall=False,
    ...     intersections=2,
    ... )
    >>> for slice_name, slice_func in slice_spec.slices():
    ...     print(slice_name)
    ...     # do something with slice_func here (e.g. dataset.filter(slice_func))
    feature_1:non_null
    feature_2:non_null
    feature_1:non_null&feature_2:non_null

    """  # noqa: W505

    spec_list: List[Dict[str, Dict[str, Any]]] = field(
        default_factory=lambda: [{}],
        init=True,
        repr=True,
        hash=True,
        compare=True,
    )
    intersections: Optional[Union[List[Tuple[int, ...]], int]] = None
    validate: bool = True
    include_overall: bool = True
    column_names: Optional[List[str]] = None

    _registry: Dict[str, Callable[[Dict[str, Any]], List[bool]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Create and register slice functions out of the slice specifications."""
        self.spec_list = copy.deepcopy(self.spec_list)
        if self.intersections is not None:
            self._create_intersections()
        for slice_spec in self.spec_list:
            self._parse_and_register_slice_specs(slice_spec)

        if self.include_overall:
            self._registry["overall"] = overall

    def add_slice_spec(self, slice_spec: Dict[str, Dict[str, Any]]) -> None:
        """Add slice specification to the list of slice specifications.

        Parameters
        ----------
        slice_spec : Dict[str, Dict[str, Any]]
            A dictionary mapping column names to dictionaries containing one or more of
            the following keys: `value`, `min_value`, `max_value`, `year`, `month`,
            `day`, `hour`, `negate`, `keep_nulls`. See :class:`SliceSpec` for more
            details on the slice specification format.

        """
        self._parse_and_register_slice_specs(slice_spec=slice_spec)
        self.spec_list.append(slice_spec)

    def get_slices(
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], List[bool]]]:
        """Return the slice function registry."""
        return self._registry

    def slices(self) -> Generator[Tuple[str, Callable[..., Any]], None, None]:
        """Return a generator of slice names and slice functions."""
        for registration_key, slice_function in self._registry.items():
            yield registration_key, slice_function

    def _create_intersections(self) -> None:
        """Create intersections of slices."""
        intersect_list = []
        if isinstance(self.intersections, list) and isinstance(
            self.intersections[0], tuple
        ):
            for intersection in self.intersections:
                intersect_dict = {}
                for index in set(intersection):
                    intersect_dict.update(self.spec_list[index])
                intersect_list.append(intersect_dict)
        elif isinstance(self.intersections, int):
            combinations = itertools.combinations(self.spec_list, self.intersections)
            for combination in combinations:
                intersect_dict = {}
                for slice_ in combination:
                    intersect_dict.update(slice_)
                intersect_list.append(intersect_dict)
        else:
            raise ValueError(
                "Expected `intersections` to be a list of tuples or an integer. "
                f"Got {self.intersections} instead.",
            )
        self.spec_list.extend(intersect_list)

        # remove duplicates
        seen = set()
        result = []

        for spec in self.spec_list:
            spec_str = json.dumps(spec, sort_keys=True)
            if spec_str not in seen:
                seen.add(spec_str)
                result.append(spec)

        seen.clear()
        self.spec_list = result

    def _parse_and_register_slice_specs(
        self,
        slice_spec: Dict[str, Dict[str, Any]],
    ) -> None:
        """Construct and register a slice functions from slice specifications."""
        if not isinstance(slice_spec, dict):
            raise TypeError(
                f"Expected `slice_spec` to be a dictionary. Got {type(slice_spec)}",
            )

        if len(slice_spec) == 0:  # empty dictionary. Interpret as `overall` slice
            registration_key = "overall"
            slice_function = overall
        elif len(slice_spec) == 1:  # slice on a single feature
            registration_key, slice_function = self._parse_single_spec_dict(slice_spec)
        else:  # compound slicing (bitwise AND of component slices)
            registration_key = ""
            slice_functions = []
            for column_name, spec in slice_spec.items():
                sub_registration_key, slice_function = self._parse_single_spec_dict(
                    {column_name: spec},
                )
                slice_functions.append(slice_function)
                registration_key += f"{sub_registration_key}&"
            registration_key = registration_key[:-1]  # remove trailing ampersand

            slice_function = partial(compound_filter, slice_functions=slice_functions)

        self._registry[registration_key] = slice_function

    def _parse_single_spec_dict(
        self,
        slice_spec: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Callable[..., List[bool]]]:
        """Return the registration key and slice function for a single slice spec."""
        column_name, spec = next(iter(slice_spec.items()))

        # validate column name and spec
        self._check_column_names(column_names=column_name)
        if not isinstance(spec, dict):
            raise TypeError(
                f"Expected feature value to be a dictionary. Got {type(spec)} ",
            )

        if "value" in spec:  # filter on exact value
            substring: Union[Any, List[Any]] = spec["value"]
            negated: bool = spec.get("negate", False)

            registration_key = f"{column_name}:{substring}"
            if isinstance(substring, list):
                # show at most 10 values in registration key. If more than 10,
                # show first 5, ..., last 5
                if len(substring) > 10:
                    value_list_repr = (
                        ", ".join(map(str, substring[:5]))
                        + ", ..., "
                        + ", ".join(map(str, substring[-5:]))
                    )
                else:
                    value_list_repr = ", ".join(map(str, substring))

                registration_key = f"{column_name}:{value_list_repr}"

            slice_function = partial(
                filter_value,
                column_name=column_name,
                value=substring,
                negate=negated,
                keep_nulls=spec.get("keep_nulls", False),
            )
        elif "min_value" in spec or "max_value" in spec:
            min_value = spec.get("min_value", -np.inf)
            max_value = spec.get("max_value", np.inf)
            min_inclusive = spec.get("min_inclusive", True)
            max_inclusive = spec.get("max_inclusive", True)
            negated = spec.get("negate", False)

            min_end = "[" if min_inclusive else "("
            max_end = "]" if max_inclusive else ")"
            registration_key = (
                f"{column_name}:{min_end}{min_value} - {max_value}{max_end}"
            )

            slice_function = partial(
                filter_range,
                column_name=column_name,
                min_value=min_value,
                max_value=max_value,
                min_inclusive=min_inclusive,
                max_inclusive=max_inclusive,
                negate=negated,
                keep_nulls=spec.get("keep_nulls", False),
            )
        elif any(k in spec for k in ("year", "month", "day", "hour")):
            year = spec.get("year")
            month = spec.get("month")
            day = spec.get("day")
            hour = spec.get("hour")
            negated = spec.get("negate", False)

            # create registration key with year, month, day, hour if specified
            registration_key = f"{column_name}:" + ", ".join(
                [
                    f"{k}={v}"
                    for k, v in zip(
                        ("year", "month", "day", "hour"),
                        (year, month, day, hour),
                    )
                    if v is not None
                ],
            )

            slice_function = partial(
                filter_datetime,
                column_name=column_name,
                year=year,
                month=month,
                day=day,
                hour=hour,
                negate=negated,
                keep_nulls=spec.get("keep_nulls", False),
            )
        elif "contains" in spec:
            substring = spec["contains"]
            negated = spec.get("negate", False)

            registration_key = f"{column_name}:contains {substring}"

            slice_function = partial(
                filter_string_contains,
                column_name=column_name,
                contains=substring,
                negate=negated,
                keep_nulls=spec.get("keep_nulls", False),
            )
        elif "keep_nulls" in spec:
            keep_nulls = spec["keep_nulls"]
            negated = spec.get("negate", False)

            # keep_nulls=True and negate=True => filter_non-null(negate=False)
            # keep_nulls=False and negate=True => filter_non-null(negate=True)
            # keep_nulls=True and negate=False => filter_non-null(negate=True)
            # keep_nulls=False and negate=False => filter_non-null(negate=False)
            negated = keep_nulls ^ negated  # XOR

            registration_key = f"{column_name}:non_null"
            slice_function = partial(
                filter_non_null,
                column_names=column_name,
                negate=negated,
            )
        else:
            raise ValueError(
                "Expected the slice specification to contain `value`, `min_value`, "
                "`max_value`, `contains`, `year`, `month`, `day`, `hour` or "
                f"`keep_nulls`. Got {spec} instead.",
            )

        if negated:
            registration_key = f"!({registration_key})"

        return registration_key, slice_function

    def _check_column_names(self, column_names: Union[str, List[str]]) -> None:
        """Check that the given column names are valid."""
        if isinstance(column_names, list):
            for column_name in column_names:
                self._check_column_names(column_name)

        if isinstance(column_names, str):
            if (
                self.validate
                and self.column_names is not None
                and column_names not in self.column_names
            ):
                raise KeyError(
                    f"Column name '{column_names}' is not in the dataset. "
                    f"Valid column names are: {self.column_names}",
                )
        else:
            raise TypeError(
                "Expected `column_names` to be a string or list of strings."
                f"Got {type(column_names)} instead.",
            )


# filter functions
def overall(examples: Union[pa.Table, LazyBatch]) -> List[bool]:
    """Return True for all examples.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A batch of examples.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples.

    """
    _check_examples(examples)
    return [True] * (
        len(list(examples.values())[0])
        if isinstance(examples, LazyBatch)
        else len(examples)
    )


def filter_non_null(
    examples: Union[pa.Table, LazyBatch],
    column_names: Union[str, List[str]],
    negate: bool = False,
) -> List[bool]:
    """Return True for all examples where the feature/column is not null.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A batch of examples to filter.
    column_names : Union[str, List[str]]
        The column name(s) on which to filter.
    negate : bool, optional, default=False
        If `True`, negate the filter, i.e. return `True` for all examples where
        the value is null.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples where the value is
        not null.

    Notes
    -----
    Floating-point NaN values will not be considered as null.

    """
    _check_examples(examples)
    if not (
        isinstance(column_names, str)
        or (
            isinstance(column_names, list)
            and all(isinstance(key, str) for key in column_names)
        )
    ):
        raise ValueError(
            "Expected `column_names` to be a string or list of strings. "
            f"Got {column_names} of type {type(column_names)}",
        )

    if isinstance(column_names, str):
        column_names = [column_names]

    mask = pc.invert(pc.is_null(examples[column_names[0]]))
    for column_name in column_names[1:]:
        mask = pc.and_not(mask, pc.is_null(examples[column_name]))

    if negate:
        mask = pc.invert(mask)

    return mask.to_pylist()  # type: ignore


def filter_value(
    examples: Union[pa.Table, LazyBatch],
    column_name: str,
    value: Union[Any, List[Any]],
    negate: bool = False,
    keep_nulls: bool = False,
) -> List[bool]:
    """Return True for all examples where the feature/column has the given value.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A batch of examples to filter.
    column_name : str
        The column name on which to filter.
    value : Union[Any, List[Any]]
        The value or values to find. Exact match is performed.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples where the column does not have
        the given value.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples in the column where the value is null.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples where the feature
        has the given value or values.

    """
    _check_examples(examples)
    value_is_datetime = is_datetime(value)  # only checks timestrings

    if not isinstance(value, list):
        value = [value]
    value_arr: pa.Array = pa.array(value)

    if value_is_datetime:
        value_arr = pc.cast(value_arr, pa.timestamp("ns"))

    example_values = (
        pc.cast(examples[column_name], pa.timestamp("ns"))
        if value_is_datetime
        else examples[column_name]
    )

    mask = pc.is_in(example_values, value_arr)

    if negate:
        mask = pc.invert(mask)

    nulls = pc.is_null(example_values)
    mask = pc.or_(mask, nulls) if keep_nulls else pc.and_not(mask, nulls)

    return mask.to_pylist()  # type: ignore


def filter_range(
    examples: Union[pa.Table, LazyBatch],
    column_name: str,
    min_value: float = -np.inf,
    max_value: float = np.inf,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    negate: bool = False,
    keep_nulls: bool = False,
) -> List[bool]:
    """Return True for all examples where the value is in the given range.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A batch of examples to filter.
    column_name : str
        The column name on which to filter.
    min_value : float, optional, default=-np.inf
        The minimum value of the range.
    max_value : float, optional, default=np.inf
        The maximum value of the range.
    min_inclusive : bool, optional, default=True
        If `True`, include the minimum value in the range.
    max_inclusive : bool, optional, default=True
        If `True`, include the maximum value in the range.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples in the column where the value is
        not in the given range.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples in the column where the value is null.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples in the column where
        the value is in the given range.

    Raises
    ------
    ValueError
        If `max_value` is less than `min_value` or if `min_value` and `max_value`
        are equal and either `min_inclusive` or `max_inclusive` is False.
    TypeError
        If the column does not contain numeric or datetime values.

    """
    _check_examples(examples)
    # handle datetime values
    min_value, max_value, value_is_datetime = _maybe_convert_to_datetime(
        min_value,
        max_value,
    )

    if min_value > max_value:
        raise ValueError(
            "Expected `min_value` to be less than or equal to `max_value`, but got "
            f"min_value={min_value} and max_value={max_value}.",
        )
    if min_value == max_value and not (min_inclusive and max_inclusive):
        raise ValueError(
            "`min_value` and `max_value` are equal and either `min_inclusive` or "
            "`max_inclusive` is False. This would result in an empty range.",
        )

    example_values = pa.array(examples[column_name])
    if value_is_datetime:
        example_values = pc.cast(example_values, pa.timestamp("ns"))
        min_value = np.repeat(min_value, len(example_values))  # type: ignore[assignment]
        max_value = np.repeat(max_value, len(example_values))  # type: ignore[assignment]

    if not (  # column does not contain number or datetime values
        pa.types.is_integer(example_values.type)
        or pa.types.is_floating(example_values.type)
        or pa.types.is_timestamp(example_values.type)
    ):
        raise TypeError(
            "Expected feature to be numeric or datetime, but got "
            f"{example_values.type}.",
        )

    ge = (
        pc.greater_equal(example_values, min_value)
        if min_inclusive
        else pc.greater(example_values, min_value)
    )
    le = (
        pc.less_equal(example_values, max_value)
        if max_inclusive
        else pc.less(example_values, max_value)
    )

    mask = pc.and_(ge, le).fill_null(False)

    if negate:
        mask = pc.invert(mask)

    nulls = pc.is_null(example_values)
    mask = pc.or_(mask, nulls) if keep_nulls else pc.and_not(mask, nulls)

    return mask.to_pylist()  # type: ignore


def filter_datetime(
    examples: Union[pa.Table, LazyBatch],
    column_name: str,
    year: Optional[Union[int, str, List[int], List[str]]] = None,
    month: Optional[Union[int, List[int]]] = None,
    day: Optional[Union[int, List[int]]] = None,
    hour: Optional[Union[int, List[int]]] = None,
    negate: bool = False,
    keep_nulls: bool = False,
) -> List[bool]:
    """Return True for all examples where the datetime value matches the given datetime.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A batch of examples to filter.
    column_name : str
        The column name on which to filter.
    year : int, str, List[int], List[str], optional, default=None
        The year to match. If string, it must be a valid year string (e.g. "2020").
        If a list is provided, return `True` for all examples where the year matches
        any of the values in the list.
    month : int, List[int], optional, default=None
        The month to match. If a list is provided, return `True` for all examples
        where the month matches any of the values in the list.
    day : int, List[int], optional, default=None
        The day to match. If a list is provided, return `True` for all examples
        where the day matches any of the values in the list.
    hour : int, List[int], optional, default=None
        The hour to match. If a list is provided, return `True` for all examples
        where the hour matches any of the values in the list.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples where the value does not match
        the given datetime components.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples that have a null value.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples where the value of
        a column matches the given datetime components.

    Raises
    ------
    TypeError
        If the column does not contain datetime values.

    """
    _check_examples(examples)
    example_values = pa.array(examples[column_name])
    try:
        example_values = pc.cast(example_values, pa.timestamp("ns"))
    except ArrowInvalid as exc:
        raise TypeError(
            "Expected datetime feature, but got feature of type "
            f"{example_values.dtype.name}.",
        ) from exc

    def _apply_mask(
        values: pa.Int64Array,
        value_set: Union[int, str, List[int], List[str]],
        mask: pa.BooleanArray,
    ) -> pa.BooleanArray:
        if isinstance(value_set, (str, int)):
            value_set = [value_set]  # type: ignore[assignment]

        return pc.and_(
            mask,
            pc.is_in(
                values,
                pa.array(np.asanyarray(value_set, dtype=int), type=pa.int64()),
            ),
        )

    mask = pa.array([True] * len(example_values), type=pa.bool_())
    if year is not None:
        years = pc.year(example_values)
        mask = _apply_mask(years, year, mask)
    if month is not None:
        months = pc.month(example_values)
        mask = _apply_mask(months, month, mask)
    if day is not None:
        days = pc.year(example_values)
        mask = _apply_mask(days, day, mask)
    if hour is not None:
        hours = pc.hour(example_values)
        mask = _apply_mask(hours, hour, mask)

    if negate:
        mask = pc.invert(mask)

    nulls = pc.is_null(example_values)
    mask = pc.or_(mask, nulls) if keep_nulls else pc.and_not(mask, nulls)

    return mask.to_pylist()  # type: ignore


def filter_string_contains(
    examples: Union[pa.Table, LazyBatch],
    column_name: str,
    contains: Union[str, List[str]],
    negate: bool = False,
    keep_nulls: bool = False,
) -> List[bool]:
    """Return True for all examples where the value contains the given substring.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A batch of examples to filter.
    column_name : str
        The column name on which to filter.
    contains : str, List[str]
        The substring(s) to match. If a list is provided, return `True` for all
        examples where the value contains any of the substrings in the list.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples where the value does not contain
        the given substring.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples that have a null value.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples where the value of
        a column contains the given substring.

    Raises
    ------
    TypeError
        If the column does not contain string values or if the values in
        `contains` are not strings.

    """
    _check_examples(examples)
    # make sure the column has string type
    example_values = pa.array(examples[column_name])
    if not pa.types.is_string(example_values.type):
        raise ValueError(
            "Expected string feature, but got feature of type "
            f"{example_values.type}.",
        )

    # get all the values that contain the given substring
    mask = pa.array([False] * len(example_values), type=pa.bool_())
    if isinstance(contains, str):
        contains = [contains]

    for substring in contains:
        if not isinstance(substring, str):
            raise TypeError(
                f"Expected string value for `contains`, but got value of type "
                f"{type(substring)}.",
            )
        mask = pc.or_(mask, pc.match_substring(example_values, substring))

    if negate:
        mask = pc.invert(mask)

    nulls = pc.is_null(example_values)
    mask = pc.or_(mask, nulls) if keep_nulls else pc.and_not(mask, nulls)

    return mask.to_pylist()  # type: ignore


def compound_filter(
    examples: Union[pa.Table, LazyBatch],
    slice_functions: List[Callable[..., List[bool]]],
) -> List[bool]:
    """Combine the result of multiple slices using bitwise AND.

    Parameters
    ----------
    examples : pyarrow.Table, datasets.formatting.formatting.LazyBatch
        A dictionary mapping column names to values.
    slice_functions : List[Callable[..., List[bool]]]
        A list of functions to apply to the examples. The signature of each
        function should be: `slice_function(examples, **kwargs)`. The result of
        each function should be a list of booleans.

    Returns
    -------
    List[bool]
        A list of booleans containing `True` for all examples where each slice
        function returns `True`.

    """
    _check_examples(examples)
    mask: List[bool] = np.bitwise_and.reduce(
        [slice_function(examples) for slice_function in slice_functions],
    )

    return mask


# utility functions
def _check_examples(examples: Union[pa.Table, LazyBatch]) -> None:
    """Check the type of `examples."""
    if not isinstance(examples, (pa.Table, LazyBatch)):
        raise TypeError(
            "Expected `examples` to be an instance of pyarrow.table or "
            "datasets.formatting.formatting.LazyBatch but got "
            f"{type(examples)}"
        )


def is_datetime(
    value: Union[
        str,
        datetime.datetime,
        np.datetime64,
        np.ndarray[Any, np.dtype[Any]],
        List[Any],
        Any,
    ],
) -> bool:
    """Check if the given value is a datetime.

    Parameters
    ----------
    value : Union[str, datetime.datetime, np.datetime64, np.ndarray, List]
        The value(s) to check.

    Returns
    -------
    bool
        True if the value is a datetime, False otherwise.

    """
    if isinstance(value, str):
        try:
            parse(value)
            return True
        except ValueError:
            return False
    if isinstance(value, (list, np.ndarray)):
        return all((is_datetime(v) for v in value))
    if isinstance(value, (datetime.datetime, np.datetime64)):  # noqa: SIM103
        return True

    return False


def _maybe_convert_to_datetime(min_value: Any, max_value: Any) -> Tuple[Any, Any, bool]:
    """Convert datetime and infinity values to np.datetime64.

    Parameters
    ----------
    min_value : Any
        The minimum value.
    max_value : Any
        The maximum value.

    Returns
    -------
    Tuple[Any, Any, bool]
        The minimum and maximum values, and a boolean indicating whether the
        values are datetime values.

    """
    if isinstance(min_value, datetime.date):
        min_value = datetime.datetime.combine(min_value, datetime.time.min)
    if isinstance(max_value, datetime.date):
        max_value = datetime.datetime.combine(max_value, datetime.time.max)

    # convert datetime and infinity values to np.datetime64
    value_is_datetime = False
    if is_datetime(min_value):
        min_value = np.datetime64(min_value)
        value_is_datetime = True
        if max_value == np.inf:
            max_value = pd.Timestamp.max.to_datetime64()
    if is_datetime(max_value):
        max_value = np.datetime64(max_value)
        value_is_datetime = True
        if min_value == -np.inf:
            min_value = pd.Timestamp.min.to_datetime64()

    return min_value, max_value, value_is_datetime
