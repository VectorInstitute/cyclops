"""Functions and classes for creating subsets of Hugging Face datasets."""

import datetime
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.parser import parse


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
    validate : bool, default=True
        Whether to validate the column names in the slice specifications.
    include_overall : bool, default=True
        Whether to include an `overall` slice that selects all examples.
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
    ...         {"feature_1": {
    ...             "min_value": 5,
    ...             "max_value": 60,
    ...             "min_inclusive": False,
    ...             "max_inclusive": False}
    ...         },
    ...         {"feature_1": {"year": [2020, 2021, 2022]}},
    ...         {"feature_1": {"month": [6, 7, 8]}},
    ...         {"feature_1": {"month": 6, "day": 1}},
    ...         {"feature_1": {"contains": "value_1"}},
    ...         {"feature_1": {"contains": ["value_1", "value_2"]}},
    ...         {
    ...             "feature_1": {"value": "value_1"},
    ...             "feature_2": {
    ...                 "min_value": "2020-01-01", keep_nulls: False,
    ...             },
    ...             "feature_3": {"year": ["2000", "2010", "2020"]},
    ...         },
    ...     ],
    ... )
    >>> for slice_name, slice_func in slice_spec.slices():
    ...     print(slice_name)
    ...     # do something with slice_func here (e.g. dataset.filter(slice_func))

    """

    spec_list: List[Dict[str, Dict[str, Any]]] = field(
        default_factory=lambda: [{}],
        init=True,
        repr=True,
        hash=True,
        compare=True,
    )
    validate: bool = True
    include_overall: bool = True
    column_names: Optional[List[str]] = None

    _registry: Dict[str, Callable[[Dict[str, Any]], Union[bool, List[bool]]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Create and register slice functions out of the slice specifications."""
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
    ) -> Dict[str, Callable[[Dict[str, Any]], Union[bool, List[bool]]]]:
        """Return the slice function registry."""
        return self._registry

    def slices(self) -> Generator[Tuple[str, Callable[..., Any]], None, None]:
        """Return a generator of slice names and slice functions."""
        for registration_key, slice_function in self._registry.items():
            yield registration_key, slice_function

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
    ) -> Tuple[str, Callable[..., Union[bool, List[bool]]]]:
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
def overall(examples: Dict[str, Any]) -> Union[bool, List[bool]]:
    """Return True for all examples.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples.

    """
    result: List[bool] = np.ones_like(
        next(iter(examples.values())),
        dtype=bool,
    ).tolist()
    if len(result) == 1:
        return result[0]
    return result


def filter_non_null(
    examples: Dict[str, Any],
    column_names: Union[str, List[str]],
    negate: bool = False,
) -> Union[bool, List[bool]]:
    """Return True for all examples where the feature/column is not null.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
    column_names : Union[str, List[str]]
        The column name(s) on which to filter.
    negate : bool, optional, default=False
        If `True`, negate the filter, i.e. return `True` for all examples where
        the value is null/nan.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the value is not null/nan.

    """
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

    result = pd.notnull(examples[column_names[0]])
    for column_name in column_names[1:]:
        result &= pd.notnull(examples[column_name])

    if negate:
        result = ~result

    return result.tolist()  # type: ignore


def filter_value(
    examples: Dict[str, Any],
    column_name: str,
    value: Union[Any, List[Any]],
    negate: bool = False,
    keep_nulls: bool = False,
) -> Union[bool, List[bool]]:
    """Return True for all examples where the feature/column has the given value.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
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
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the feature has the given value or values.

    """
    value_is_datetime = is_datetime(value)  # only checks timestrings

    value = pd.Series(
        value,
        dtype="datetime64[ns]" if value_is_datetime else None,
    ).to_numpy()

    example_values = pd.Series(
        examples[column_name],
        dtype="datetime64[ns]" if value_is_datetime else None,
    ).to_numpy()

    result = np.isin(example_values, value, invert=negate)

    if keep_nulls:
        result |= pd.isnull(example_values)
    else:
        result &= pd.notnull(example_values)

    return result.tolist()  # type: ignore


def filter_range(
    examples: Dict[str, Any],
    column_name: str,
    min_value: float = -np.inf,
    max_value: float = np.inf,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    negate: bool = False,
    keep_nulls: bool = False,
) -> Union[bool, List[bool]]:
    """Return True for all examples where the value is in the given range.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
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
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples in the
        column where the value is in the given range.

    Raises
    ------
    ValueError
        If `max_value` is less than `min_value` or if `min_value` and `max_value`
        are equal and either `min_inclusive` or `max_inclusive` is False.
    TypeError
        If the column does not contain numeric or datetime values.

    """
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

    example_values = pd.Series(
        examples[column_name],
        dtype="datetime64[ns]" if value_is_datetime else None,
    ).to_numpy()

    if not (  # column does not contain number or datetime values
        np.issubdtype(example_values.dtype, np.number)
        or np.issubdtype(example_values.dtype, np.datetime64)
    ):
        raise TypeError(
            "Expected feature to be numeric or datetime, but got "
            f"{example_values.dtype}.",
        )

    result = (
        ((example_values > min_value) & (example_values < max_value))
        | ((example_values == min_value) & min_inclusive)
        | ((example_values == max_value) & max_inclusive)
    )

    if negate:
        result = ~result

    if keep_nulls:
        result |= pd.isnull(example_values)
    else:
        result &= pd.notnull(example_values)

    return result.tolist()  # type: ignore


def filter_datetime(
    examples: Dict[str, Any],
    column_name: str,
    year: Optional[Union[int, str, List[int], List[str]]] = None,
    month: Optional[Union[int, List[int]]] = None,
    day: Optional[Union[int, List[int]]] = None,
    hour: Optional[Union[int, List[int]]] = None,
    negate: bool = False,
    keep_nulls: bool = False,
) -> Union[bool, List[bool]]:
    """Return True for all examples where the datetime value matches the given datetime.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
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
        If `True`, return `True` for all examples that have a null/nan/NaT value.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the value of a column matches the given datetime components.

    Raises
    ------
    TypeError
        If the column does not contain datetime values.

    """
    # make sure the column has datetime type
    example_values = pd.Series(examples[column_name]).to_numpy()
    try:
        example_values = example_values.astype("datetime64")
    except ValueError as exc:
        raise TypeError(
            "Expected datetime feature, but got feature of type "
            f"{example_values.dtype.name}.",
        ) from exc

    # convert the datetime values to year, month, day, hour
    years, months, days = [
        example_values.astype(f"M8[{unit}]") for unit in ["Y", "M", "D"]
    ]

    # get all the values that match the given datetime
    # acknowledgement: https://stackoverflow.com/a/56260054
    result = np.ones_like(example_values, dtype=bool)
    if year is not None:
        result &= np.isin(
            element=years.astype(int) + 1970,
            test_elements=np.asanyarray(year, dtype=int),
        )
    if month is not None:
        result &= np.isin(
            element=((months - years) + 1).astype(int),
            test_elements=np.asanyarray(month, dtype=int),
        )
    if day is not None:
        result &= np.isin(
            element=((days - months) + 1).astype(int),
            test_elements=np.asanyarray(day, dtype=int),
        )
    if hour is not None:
        result &= np.isin(
            element=(example_values - days).astype("m8[h]").astype(int),
            test_elements=np.asanyarray(hour, dtype=int),
        )

    if negate:
        result = ~result

    if keep_nulls:
        result |= pd.isnull(example_values)
    else:
        result &= pd.notnull(example_values)

    return result.tolist()  # type: ignore


def filter_string_contains(
    examples: Dict[str, Any],
    column_name: str,
    contains: Union[str, List[str]],
    negate: bool = False,
    keep_nulls: bool = False,
) -> Union[bool, List[bool]]:
    """Return True for all examples where the value contains the given substring.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
    column_name : str
        The column name on which to filter.
    contains : str, List[str]
        The substring to match. If a list is provided, return `True` for all
        examples where the value contains any of the substrings in the list.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples where the value does not contain
        the given substring.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples that have a null value.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the value of a column contains the given substring.

    Raises
    ------
    TypeError
        If the column does not contain string values or if the values in
        `contains` are not strings.

    """
    # make sure the column has string type
    example_values = pd.Series(examples[column_name])
    if example_values.dtype.name != "object" and not isinstance(
        example_values.dtype,
        pd.StringDtype,
    ):
        raise ValueError(
            "Expected string feature, but got feature of type "
            f"{example_values.dtype.name}.",
        )
    if example_values.dtype.name == "object":  # object type could be string
        try:
            _ = example_values.str
        except AttributeError as exc:
            raise TypeError(
                "Expected string feature, but got feature of type "
                f"{example_values.dtype.name}.",
            ) from exc

    # get all the values that contain the given substring
    result = np.zeros_like(example_values, dtype=bool)
    if isinstance(contains, str):
        contains = [contains]

    for substring in contains:
        if not isinstance(substring, str):
            raise TypeError(
                f"Expected string value for `contains`, but got value of type "
                f"{type(substring)}.",
            )
        result |= example_values.str.contains(substring, case=False).to_numpy(
            dtype=bool,
        )

    if negate:
        result = ~result

    if keep_nulls:
        result |= pd.isnull(example_values)
    else:
        result &= pd.notnull(example_values)

    return result.tolist()  # type: ignore


def compound_filter(
    examples: Dict[str, Any],
    slice_functions: List[Callable[..., Union[bool, List[bool]]]],
) -> Union[bool, List[bool]]:
    """Combine the result of multiple slices using bitwise AND.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary mapping column names to values.
    slice_functions : List[Callable]
        A list of functions to apply to the examples. The signature of each
        function should be: `slice_function(examples, **kwargs)`.
        The result of each function should be a boolean or a list of booleans.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        each slice function returns `True`.

    """
    result: Union[bool, List[bool]] = np.bitwise_and.reduce(
        [slice_function(examples) for slice_function in slice_functions],
    )

    return result


# utility functions
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
    if isinstance(value, (datetime.datetime, np.datetime64)):
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
