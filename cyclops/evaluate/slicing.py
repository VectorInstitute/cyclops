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
    feature_keys : List[str]
        List of feature keys to slice on. Each key can be a single feature or a list of
        features. The slice selects non-null values for the feature(s).
    feature_values : List[Union[Mapping[str, Any], List[Mapping[str, Any]]]]
        List of feature values to slice on. Each value is a dictionary mapping a feature
        key to a dictionary of feature value specifications. The slice selects rows
        where the feature value matches the specification. The feature value
        specification can be a single value, a range, or a list of values.
        For a single value or list of values, the specification expects a "value" key
        with the value or list of values. For a range, the specification expects a "min"
        and "max" key with the minimum and maximum values. Optionally, "min_inclusive"
        and "max_inclusive" keys can be set to False to exclude the minimum and maximum
        values from the range. The specification can also specify whether to discard
        null values, in which case "keep_null" is expected to be set to False. The
        specification can also be negated, in which case the slice selects rows where
        the feature value does not match the specification. This is done by setting the
        "negate" key to True. If the dictionary has more than one key, the slice selects
        rows where the feature value matches all specifications.
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
    column_names : Optional[List[str]]
        List of column names in the data.
    validate : bool
        Whether to validate the feature keys and values.
    _slice_function_registry : Dict[str, Callable]
        Dictionary mapping slice function registration keys to slice functions.

    """

    feature_keys: List[Union[str, List[str]]] = field(
        default_factory=list, init=True, repr=True, hash=True, compare=True
    )
    feature_values: List[Union[Mapping[str, Any], List[Mapping[str, Any]]]] = field(
        default_factory=lambda: [{}], init=True, repr=True, hash=True, compare=True
    )
    column_names: Optional[List[str]] = None
    validate: bool = True
    _slice_function_registry: Dict[str, Callable] = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self):
        """Parse the slice definitions and construct the slice functions."""
        for feature_key in self.feature_keys:
            self._parse_feature_keys(feature_key)

        for slice_def in self.feature_values:
            self._parse_feature_values(slice_def)

    def _check_feature_keys(self, keys: Union[str, List[str]]):
        if isinstance(keys, list):
            for key in keys:
                self._check_feature_keys(key)

        if self.validate and self.column_names is not None:
            if isinstance(keys, str) and keys not in self.column_names:
                raise KeyError(f"{keys} is not a valid column name")

    def _parse_feature_keys(self, feature_key: Union[str, List[str]]) -> None:
        self._check_feature_keys(keys=feature_key)
        if isinstance(feature_key, str):
            registration_key = f"slice_non_null:{feature_key}"
            self._slice_function_registry[registration_key] = partial(
                self.slice_non_null, feature_key=feature_key
            )
        elif isinstance(feature_key, list):
            registration_key = f"slice_non_null_list:{feature_key}"
            self._slice_function_registry[registration_key] = partial(
                self.slice_non_null_list, feature_list=feature_key
            )

    def _parse_single_feature_value_dict(
        self, slice_def: Mapping[str, Mapping]
    ) -> Tuple[str, Callable]:
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
                if negated:
                    value_list_repr = f"!({value_list_repr})"

                registration_key = f"{feature_key}:{value_list_repr}"
                slice_function = partial(
                    self.slice_feature_value_list,
                    feature_key=feature_key,
                    value_list=value,
                    negate=negated,
                    keep_nulls=feature_value.get("keep_nulls", True),
                )
            else:
                registration_key = f"{feature_key}:{value}"
                if negated:
                    registration_key = f"!({registration_key})"

                slice_function = partial(
                    self.slice_feature_value,
                    feature_key=feature_key,
                    value=value,
                    negate=negated,
                    keep_nulls=feature_value.get("keep_nulls", True),
                )

        elif "min" in feature_value or "max" in feature_value:
            min_value = feature_value.get("min", float("-inf"))
            max_value = feature_value.get("max", float("inf"))
            registration_key = f"{feature_key}:{min_value} - {max_value}"

            negated = feature_value.get("negate", False)
            if negated:
                registration_key = f"!({registration_key})"

            slice_function = partial(
                self.slice_feature_value_range,
                feature_key=feature_key,
                min=min_value,
                max=max_value,
                min_inclusive=feature_value.get("min_inclusive", True),
                max_inclusive=feature_value.get("max_inclusive", False),
                negate=negated,
                keep_nulls=feature_value.get("keep_nulls", True),
            )
        else:
            raise ValueError(f"Invalid `feature_value` specification: {slice_def}")

        return registration_key, slice_function

    def _parse_feature_values(self, slice_def: Mapping[str, Mapping]) -> None:
        if not isinstance(slice_def, Mapping):
            raise ValueError(f"Invalid `feature_value` specification: {slice_def}")

        if len(slice_def) == 0:  # empty dict - interpret as `slice_all`
            registration_key = "slice_all"
            slice_function = self.slice_all
        elif len(slice_def) == 1:  # single feature
            registration_key, slice_function = self._parse_single_feature_value_dict(
                slice_def
            )
        else:  # compound slicing (bitwise AND)
            registration_key = "slice_compound_feature_value:"
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
                self.slice_compound_feature_value, slice_functions=slice_functions
            )

        self._slice_function_registry[registration_key] = slice_function

    def add_feature_keys(self, feature_keys: Union[str, List[str]]) -> None:
        """Add feature keys to the slice specification."""
        self.feature_keys.append(feature_keys)
        self._parse_feature_keys(feature_keys)

    def add_feature_values(self, feature_values: Mapping[str, Mapping]) -> None:
        """Add feature values to the slice specification."""
        self.feature_values.append(feature_values)
        self._parse_feature_values(feature_values)

    @staticmethod
    def slice_all(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
    ) -> Union[bool, List[bool]]:
        """Return True for all examples.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            A list of indices to slice.
        **kwargs
            Additional keyword arguments. Ignored.

        Returns
        -------
        Union[bool, List[bool]]
            A boolean or a list of booleans containing `True` for all examples.

        """
        result: List[bool] = [True] * len(next(iter(examples.values())))
        if len(result) == 1:
            return result[0]
        return result

    @staticmethod
    def slice_non_null(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Union[bool, List[bool]]:
        """Return True for all examples where the feature/column is not null/nan.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            The index or indices of the examples in the batch.
        **kwargs
            Additional keyword arguments. Must include `feature_key` which is the
            name of the column to check for null/nan values.

        Returns
        -------
        Union[bool, List[bool]]
            A boolean or a list of booleans containing `True` for all examples where
            the feature is not null/nan.

        """
        feature_key = kwargs["feature_key"]
        values = examples[feature_key]
        result = pd.notnull(values)

        if isinstance(result, np.ndarray):
            result = result.tolist()

        return result

    @staticmethod
    def slice_non_null_list(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Union[bool, List[bool]]:
        """Return True for all examples where the columns are not null.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            The index or indices of the examples in the batch.
        **kwargs
            Additional keyword arguments. Must include `feature_list` which is a list
            of column names to check for null/nan values.

        Returns
        -------
        Union[bool, List[bool]]
            A boolean or a list of booleans containing `True` for all examples where
            the given list of columns are not null/nan.

        """
        feature_list = kwargs["feature_list"]
        if not isinstance(feature_list, list):
            raise ValueError(
                f"Invalid `feature_key` specification: {feature_list}. "
                "Must be a list of feature keys. Use `slice_non_null` for a single "
                "feature key."
            )

        result = np.bitwise_and.reduce(
            [pd.notnull(examples[feature_key]) for feature_key in feature_list]
        )

        if isinstance(result, np.ndarray):
            result = result.tolist()

        return result

    @staticmethod
    def slice_feature_value(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Union[bool, List[bool]]:
        """Return True for all examples where the feature/column has the given value.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            The index or indices of the examples in the batch.
        **kwargs
            Additional keyword arguments. Expected keyword arguments are:
            - `feature_key`: required
                The name of the column to check for the given value.
            - `value`: required
                The value to check for in the column.
            - `negate`: optional, default=False
                If True, return True for all examples where the feature/column does
                not have the given value.
            - `keep_null`: optional, default=True
                If True, return True for all examples where the feature/column is
                null/nan.

        Returns
        -------
        Union[bool, List[bool]]
            A boolean or a list of booleans containing `True` for all examples where
            the feature has the given value.

        """
        value = kwargs["value"]
        feature_key = kwargs["feature_key"]
        negate = kwargs.get("negate", False)
        keep_null = kwargs.get("keep_null", True)

        value_is_datetime = is_datetime(value)
        if value_is_datetime:
            value = np.datetime64(value)

        result = (
            np.asanyarray(
                examples[feature_key],
                dtype=np.datetime64 if value_is_datetime else None,
            )
            == value
        )
        if not keep_null:
            result = np.bitwise_and(result, pd.notnull(examples[feature_key]))
        if negate:
            result = np.bitwise_not(result)

        if isinstance(result, np.ndarray):
            result = result.tolist()

        return result

    @staticmethod
    def slice_feature_value_list(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Union[bool, List[bool]]:
        """Return True for all examples where the value is in the given list.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            The index or indices of the examples in the batch.
        **kwargs
            Additional keyword arguments. Expected keyword arguments are:
            - `feature_key`: required
                The name of the column to check for the given values.
            - `value_list`: required
                The list of values to check for in the column.
            - `negate`: optional, default=False
                If True, return True for all examples where the value of a column
                is not in the given list.
            - `keep_null`: optional, default=True
                If True, keep examples where the value of a column is null/nan.

        Returns
        -------
        Union[bool, List[bool]]
            A boolean or a list of booleans containing `True` for all examples where
            the value of a feature is in the given list.

        """
        value_list = kwargs["value_list"]
        feature = kwargs["feature_key"]
        negate = kwargs.get("negate", False)
        keep_null = kwargs.get("keep_null", True)

        example_values = examples[feature]

        value_is_datetime = is_datetime(value_list)
        if value_is_datetime:
            value_list = np.asanyarray(value_list, dtype=np.datetime64)
            example_values = np.asanyarray(examples[feature], dtype=np.datetime64)

        result = np.isin(example_values, value_list, invert=negate)
        if not keep_null:
            result = np.bitwise_and(result, pd.notnull(example_values))

        if isinstance(result, np.ndarray):
            result = result.tolist()

        return result

    @staticmethod
    def slice_feature_value_range(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Union[bool, List[bool]]:
        """Return True for all examples where the value is in the given range.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            The index or indices of the examples in the batch.
        **kwargs
            Additional keyword arguments. Expected keyword arguments are:
            - `feature_key`: required
                The name of the column to check for the given values.
            - `min`: optional, default=-inf
                The minimum value of the range.
            - `max`: optional, default=inf
                The maximum value of the range.
            - `min_inclusive`: optional, default=True
                If True, the minimum value is included in the range.
            - `max_inclusive`: optional, default=True
                If True, the maximum value is included in the range.
            - `negate`: optional, default=False
                If True, return True for all examples where the value of a column
                is not in the given range.
            - `keep_null`: optional, default=True
                If True, keep examples where the value of a column is null/nan.

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
        feature_key = kwargs["feature_key"]
        min_value = kwargs.get("min", float("-inf"))
        max_value = kwargs.get("max", float("inf"))
        min_inclusive = kwargs.get("min_inclusive", True)
        max_inclusive = kwargs.get("max_inclusive", True)
        negate = kwargs.get("negate", False)
        keep_null = kwargs.get("keep_null", True)

        # handle datetime values
        value_is_datetime = False
        if is_datetime(min_value):
            min_value = np.datetime64(min_value)
            value_is_datetime = True
        if is_datetime(max_value):
            max_value = np.datetime64(max_value)
            value_is_datetime = True

        if value_is_datetime and min_value == float("-inf"):
            min_value = np.datetime64(datetime.datetime.min)
        if value_is_datetime and max_value == float("inf"):
            max_value = np.datetime64(datetime.datetime.max)

        if min_value > max_value:
            raise ValueError(
                f"Invalid range specification: min={min_value}, max={max_value}. "
                "min must be less than or equal to max."
            )
        if min_value == max_value and not (min_inclusive and max_inclusive):
            raise ValueError(
                f"Invalid range specification: min={min_value}, max={max_value}, "
                "min_inclusive={min_inclusive}, max_inclusive={max_inclusive}. "
                "min must be less than or equal to max."
            )

        example_values = np.asanyarray(
            examples[feature_key], dtype=np.datetime64 if value_is_datetime else None
        )

        # check that the feature is a number or datetime
        if not (
            np.issubdtype(example_values.dtype, np.number)
            or np.issubdtype(example_values.dtype, np.datetime64)
        ):
            raise ValueError(
                "Invalid range specification: "
                f"feature {feature_key} is not a number or datetime."
            )

        result = (
            ((example_values > min_value) & (example_values < max_value))
            | ((example_values == min_value) & min_inclusive)
            | ((example_values == max_value) & max_inclusive)
        )

        if not keep_null:
            result = np.bitwise_and(result, pd.notnull(example_values))
        if negate:
            result = np.bitwise_not(result)

        if isinstance(result, np.ndarray):
            result = result.tolist()

        return result

    @staticmethod
    def slice_compound_feature_value(
        examples: Dict[str, Any],
        indices: Union[int, List[int]] = None,  # pylint: disable=unused-argument
        **kwargs,
    ):
        """Combine the result of multiple slices using bitwise AND.

        Parameters
        ----------
        examples : Dict[str, Any]
            A dictionary of features and values.
        indices : Union[int, List[int]], optional, default=None
            The index or indices of the examples in the batch.
        **kwargs
            Additional keyword arguments. Expected keyword arguments are:
            - `slice_functions`: required
                A list of functions to apply to the examples. The signature of each
                function should be: `slice_function(examples, indices, **kwargs)`.
                The result of each function should be a boolean or a list of booleans.

        Returns
        -------
        Union[bool, List[bool]]
            A boolean or a list of booleans containing `True` for all examples where
            the value of a feature is in the given range.

        """
        slice_functions = kwargs["slice_functions"]

        return np.bitwise_and.reduce(
            [slice_function(examples, indices) for slice_function in slice_functions]
        )

    def get_slice_function(self, slice_key: str) -> Callable:
        """Return the slice function for the given slice key."""
        return self._slice_function_registry[slice_key]

    def get_slice_keys(self) -> List[str]:
        """Return the slice keys."""
        return list(self._slice_function_registry.keys())

    def get_slice_functions(self) -> List[Callable]:
        """Return the slice functions."""
        return list(self._slice_function_registry.values())

    def get_slices(self) -> Dict[str, Callable]:
        """Return the slice functions."""
        return self._slice_function_registry


# utility functions
def is_datetime(value: Union[str, datetime.datetime, np.datetime64, np.ndarray, List]):
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
    elif isinstance(value, (datetime.datetime, np.datetime64)):
        return True
    elif isinstance(value, (list, np.ndarray)):
        return all((is_datetime(v) for v in value))
    else:
        return False
