"""Slicing functions for evaluating model performance on subsets of the data."""
import datetime
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.parser import parse


@dataclass
class SlicingConfig:
    """Configuration for slicing the data.

    Parameters
    ----------
    feature_keys : List[str], default=[]
        List of feature keys to slice on. Each key can be a single feature or a list of
        features. The slice selects non-null values for the feature(s).
    feature_values : List[Union[Mapping[str, Any], List[Mapping[str, Any]]]]
        List of feature values to slice on. Each value is a dictionary mapping a feature
        key to a dictionary of feature value specifications. The slice selects rows
        where the feature value matches the specification. Defaults [{}] which
        selects all rows. The following feature value specifications are supported:
        - value: The feature value must match the specified value. The value can be a
          single value or a list of values. Time strings are supported.
        - min_value: The feature value must be greater than or equal to the specified
          value. Time strings are supported.
        - min_inclusive: Whether to include the minimum value in the range. If True,
          the slice selects rows where the feature value is greater than or equal to
          the minimum value.
        - max_value: The feature value must be less than or equal to the specified
          value. Time strings are supported.
        - max_inclusive: Whether to include the maximum value in the range. If True,
          the slice selects rows where the feature value is less than or equal to the
          maximum value.
        - year: The feature value must be in the specified year.
        - month: The feature value must be in the specified month.
        - day: The feature value must be in the specified day.
        - hour: The feature value must be in the specified hour.
        - negate: Whether to negate the feature value specification. If True, the slice
          selects rows where the feature value does not match the specification.
        - keep_nulls: Whether to keep rows where the feature value is null. If True,
          the slice selects rows where the feature value is null or matches the
          specification.
    column_names : Optional[List[str]], optional
        List of column names in the data. If provided, the feature keys are validated
        against the column names. If the feature keys are not valid column names, a
        KeyError is raised.
    validate : bool, default=True
        Whether to validate the feature keys and values. If True, the feature keys are
        validated against the column names.

    Attributes
    ----------
    feature_keys : List[str]
        List of feature keys to slice on.
    feature_values : List[Union[Mapping[str, Any], List[Mapping[str, Any]]]]
        List of feature values to slice on.
    column_names : List[str]
        List of column names in the data.
    validate : bool
        Whether to validate the feature keys and values.
    _slice_function_registry : Dict[str, Callable]
        Dictionary mapping slice function registration keys to slice functions.

    Examples
    --------
    >>> from cyclops.evaluate.slicing import SlicingConfig
    >>> slicing_config = SlicingConfig(
    ...     feature_keys=["feature_1", "feature_2", ["feature_3", "feature_4"]],
    ...     feature_values=[
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
    ...         {
    ...             "feature_1": {"value": "value_1"},
    ...             "feature_2": {
    ...                 "min_value": "2020-01-01", keep_nulls: False,
    ...             },
    ...             "feature_3": {"year": ["2000", "2010", "2020"]},
    ...         },
    ...     ],
    ... )

    """

    feature_keys: List[Union[str, List[str]]] = field(
        default_factory=list, init=True, repr=True, hash=True, compare=True
    )
    feature_values: List[Mapping[str, Mapping[str, Any]]] = field(
        default_factory=lambda: [{}], init=True, repr=True, hash=True, compare=True
    )
    column_names: Optional[List[str]] = None
    validate: bool = True
    _slice_function_registry: Dict[
        str, Callable[[Dict[str, Any]], Union[bool, List[bool]]]
    ] = field(default_factory=dict, init=False, repr=False, hash=False, compare=False)

    def __post_init__(self) -> None:
        """Parse the slice definitions and construct the slice functions."""
        for feature_key in self.feature_keys:
            self._parse_feature_keys(feature_key)

        for slice_def in self.feature_values:
            self._parse_feature_values(slice_def)

    def _check_feature_keys(self, keys: Union[str, List[str]]) -> None:
        """Check that the feature keys are valid."""
        if isinstance(keys, list):
            for key in keys:
                self._check_feature_keys(key)

        if self.validate and self.column_names is not None:
            if isinstance(keys, str) and keys not in self.column_names:
                raise KeyError(f"{keys} is not a valid column name")

    def _parse_feature_keys(self, feature_keys: Union[str, List[str]]) -> None:
        """Parse the feature keys and register the slice function."""
        self._check_feature_keys(keys=feature_keys)

        registration_key = f"filter_non_null:{feature_keys}"
        self._slice_function_registry[registration_key] = partial(
            filter_non_null, feature_keys=feature_keys
        )

    def _parse_single_feature_value_dict(  # pylint: disable=too-many-branches
        self, slice_def: Mapping[str, Mapping[str, Any]]
    ) -> Tuple[str, Callable[..., Union[bool, List[bool]]]]:
        """Parse a single feature value dictionary and register the slice function."""
        feature_key, feature_value = next(iter(slice_def.items()))

        # validate key and value
        if not isinstance(feature_value, Mapping):
            raise ValueError(f"Invalid `feature_value` specification: {slice_def}")
        self._check_feature_keys(keys=feature_key)

        if "value" in feature_value:
            value: Union[Any, List[Any]] = feature_value["value"]
            negated: bool = feature_value.get("negate", False)

            if isinstance(value, list):
                if len(value) <= 6:
                    value_list_repr = ",".join(map(str, value))
                else:
                    value_list_repr = (
                        ",".join(map(str, value[:3]))
                        + ",...,"
                        + ",".join(map(str, value[-3:]))
                    )

                registration_key = f"{feature_key}:{value_list_repr}"
            else:
                registration_key = f"{feature_key}:{value}"

            if negated:
                registration_key = f"!({registration_key})"

            slice_function = partial(
                filter_feature_value,
                feature_key=feature_key,
                value=value,
                negate=negated,
                keep_nulls=feature_value.get("keep_nulls", True),
            )

        elif "min_value" in feature_value or "max_value" in feature_value:
            min_value = feature_value.get("min_value", -np.inf)
            max_value = feature_value.get("max_value", np.inf)
            registration_key = f"{feature_key}:{min_value} - {max_value}"

            negated = feature_value.get("negate", False)
            if negated:
                registration_key = f"!({registration_key})"

            slice_function = partial(
                filter_feature_value_range,
                feature_key=feature_key,
                min_value=min_value,
                max_value=max_value,
                min_inclusive=feature_value.get("min_inclusive", True),
                max_inclusive=feature_value.get("max_inclusive", True),
                negate=negated,
                keep_nulls=feature_value.get("keep_nulls", True),
            )
        elif any(k in feature_value for k in ("year", "month", "day", "hour")):
            year = feature_value.get("year")
            month = feature_value.get("month")
            day = feature_value.get("day")
            hour = feature_value.get("hour")

            # create registration key with year, month, day, hour if specified
            registration_key = f"{feature_key}:"
            if year is not None:
                registration_key += f"year={year},"
            if month is not None:
                registration_key += f"month={month},"
            if day is not None:
                registration_key += f"day={day},"
            if hour is not None:
                registration_key += f"hour={hour}"

            # remove trailing comma, if any
            registration_key = registration_key.rstrip(",")

            negated = feature_value.get("negate", False)
            if negated:
                registration_key = f"!({registration_key})"

            slice_function = partial(
                filter_feature_value_datetime,
                feature_key=feature_key,
                year=year,
                month=month,
                day=day,
                hour=hour,
                negate=negated,
                keep_nulls=feature_value.get("keep_nulls", True),
            )
        else:
            raise ValueError(f"Invalid `feature_value` specification: {slice_def}")

        return registration_key, slice_function

    def _parse_feature_values(self, slice_def: Mapping[str, Mapping[str, Any]]) -> None:
        """Parse the feature values and register the slice function."""
        if not isinstance(slice_def, Mapping):
            raise ValueError(f"Invalid `feature_value` specification: {slice_def}")

        if len(slice_def) == 0:  # empty dict - interpret as `no_filter`
            registration_key = "no_filter"
            slice_function = no_filter
        elif len(slice_def) == 1:  # single feature
            registration_key, slice_function = self._parse_single_feature_value_dict(
                slice_def
            )
        else:  # compound slicing (bitwise AND)
            registration_key = "filter_compound_feature_value:"
            slice_functions = []
            for feature_key, feature_value in slice_def.items():
                (
                    sub_registration_key,
                    slice_function,
                ) = self._parse_single_feature_value_dict({feature_key: feature_value})
                slice_functions.append(slice_function)
                registration_key += f"{sub_registration_key}+"

            # remove trailing +
            registration_key = registration_key[:-1]

            # slice function is a bitwise AND of all the slice functions
            slice_function = partial(
                filter_compound_feature_value, slice_functions=slice_functions
            )

        self._slice_function_registry[registration_key] = slice_function

    def add_feature_keys(self, feature_keys: Union[str, List[str]]) -> None:
        """Add feature keys to the slice specification.

        Parameters
        ----------
        feature_keys : Union[str, List[str]]
            A feature key or a list of feature keys.

        """
        self.feature_keys.append(feature_keys)
        self._parse_feature_keys(feature_keys)

    def add_feature_values(
        self, feature_values: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Add slice definition to the configuration.

        Parameters
        ----------
        feature_values : Mapping[str, Mapping]
            A dictionary of feature keys and values.

        """
        self.feature_values.append(feature_values)
        self._parse_feature_values(feature_values)

    def get_slices(
        self,
    ) -> Dict[str, Callable[[Dict[str, Any]], Union[bool, List[bool]]]]:
        """Return the slice function registry."""
        return self._slice_function_registry


# filter functions
def no_filter(examples: Dict[str, Any]) -> Union[bool, List[bool]]:
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
    result: List[bool] = [True] * len(next(iter(examples.values())))
    if len(result) == 1:
        return result[0]
    return result


def filter_non_null(
    examples: Dict[str, Any], feature_keys: Union[str, List[str]]
) -> Union[bool, List[bool]]:
    """Return True for all examples where the feature/column is not null.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
    feature_keys : Union[str, List[str]]
        The feature key(s) or column name(s) on which to filter.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the feature(s) is not null/nan.

    """
    # make sure feature key is a string or list of strings
    if not (
        isinstance(feature_keys, str)
        or (
            isinstance(feature_keys, list)
            and all(isinstance(key, str) for key in feature_keys)
        )
    ):
        raise ValueError(
            "Expected feature_keys to be a string or list of strings. "
            f"Got: {feature_keys} of type {type(feature_keys)}"
        )

    if isinstance(feature_keys, str):
        feature_keys = [feature_keys]

    result = pd.notnull(examples[feature_keys[0]])
    for feature_key in feature_keys[1:]:
        result &= pd.notnull(examples[feature_key])

    return result.tolist()  # type: ignore


def filter_feature_value(
    examples: Dict[str, Any],
    feature_key: str,
    value: Union[Any, List[Any]],
    negate: bool = False,
    keep_nulls: bool = False,
) -> Union[bool, List[bool]]:
    """Return True for all examples where the feature/column has the given value.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
    feature_key : str
        The feature key or column name on which to find the value.
    value : Union[Any, List[Any]]
        The value or values to find. Exact match is performed.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples where the feature/column does not
        have the given value.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples where the feature/column is null.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the feature has the given value or values.

    """
    value_is_datetime = is_datetime(value)  # only checks timestrings

    value = pd.Series(
        value, dtype="datetime64[ns]" if value_is_datetime else None
    ).to_numpy()

    example_values = pd.Series(
        examples[feature_key], dtype="datetime64[ns]" if value_is_datetime else None
    ).to_numpy()

    result = np.isin(example_values, value, invert=negate)

    if keep_nulls:
        result |= pd.isnull(example_values)
    else:
        result &= pd.notnull(example_values)

    return result.tolist()  # type: ignore


def filter_feature_value_range(
    examples: Dict[str, Any],
    feature_key: str,
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
    feature_key : str
        The feature key or column name on which to filter.
    min_value : float, optional, default=-np.inf
        The minimum value of the range.
    max_value : float, optional, default=np.inf
        The maximum value of the range.
    min_inclusive : bool, optional, default=True
        If `True`, include the minimum value in the range.
    max_inclusive : bool, optional, default=True
        If `True`, include the maximum value in the range.
    negate : bool, optional, default=False
        If `True`, return `True` for all examples where the value is not in the
        given range.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples where the feature/column is null.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the value of a feature is in the given range.

    Raises
    ------
    ValueError
        If max is less than min.
    ValueError
        If min and max are equal and either min_inclusive or max_inclusive is
        False.
    ValueError
        If the feature is not numeric or datetime.

    """
    # handle datetime values
    min_value, max_value, value_is_datetime = _maybe_convert_to_datetime(
        min_value, max_value
    )

    if min_value > max_value:
        raise ValueError(
            "Expected `min_value` to be less than or equal to `max_value`, but got "
            f"`min_value={min_value}` and `max_value={max_value}`."
        )
    if min_value == max_value and not (min_inclusive and max_inclusive):
        raise ValueError(
            "`min_value` and `max_value` are equal and either `min_inclusive` or "
            "`max_inclusive` is False. This would result in an empty range."
        )

    example_values = pd.Series(
        examples[feature_key], dtype="datetime64[ns]" if value_is_datetime else None
    ).to_numpy()

    # check that the feature is a number or datetime
    if not (
        np.issubdtype(example_values.dtype, np.number)
        or np.issubdtype(example_values.dtype, np.datetime64)
    ):
        raise ValueError(
            "Expected feature to be numeric or datetime, but got "
            f"{example_values.dtype}."
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


def filter_feature_value_datetime(
    examples: Dict[str, Any],
    feature_key: str,
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
    feature_key : str
        The feature key or column name on which to filter.
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
        the given datetime.
    keep_nulls : bool, optional, default=False
        If `True`, return `True` for all examples where the feature/column is null.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the value of a feature matches the given datetime.

    Raises
    ------
    ValueError
        If the feature is not a datetime.

    """
    # make sure the column has datetime type
    example_values = pd.Series(examples[feature_key]).to_numpy()
    try:
        example_values = example_values.astype("datetime64")
    except ValueError as exc:
        raise ValueError(
            "Expected datetime feature, but got feature of type "
            f"{example_values.dtype.name}."
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


def filter_compound_feature_value(
    examples: Dict[str, Any],
    slice_functions: List[Callable[..., Union[bool, List[bool]]]],
) -> Union[bool, List[bool]]:
    """Combine the result of multiple slices using bitwise AND.

    Parameters
    ----------
    examples : Dict[str, Any]
        A dictionary of features and values.
    slice_functions : List[Callable]
        A list of functions to apply to the examples. The signature of each
        function should be: `slice_function(examples, indices, **kwargs)`.
        The result of each function should be a boolean or a list of booleans.

    Returns
    -------
    Union[bool, List[bool]]
        A boolean or a list of booleans containing `True` for all examples where
        the value of a feature is in the given range.

    """
    result: Union[bool, List[bool]] = np.bitwise_and.reduce(
        [slice_function(examples) for slice_function in slice_functions]
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
