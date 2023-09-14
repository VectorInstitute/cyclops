"""Tests for cyclops.data.slicer module."""

from functools import partial
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset
from datasets.splits import Split

import cyclops.query.ops as qo
from cyclops.data.slicer import (
    SliceSpec,
    _maybe_convert_to_datetime,
    compound_filter,
    filter_datetime,
    filter_non_null,
    filter_range,
    filter_string_contains,
    filter_value,
    overall,
)
from cyclops.query.omop import OMOPQuerier


SYNTHEA = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")


def visits_table() -> pd.DataFrame:
    """Get the visits table."""
    ops = qo.Sequential(
        qo.ConditionEquals("gender_source_value", "M"),  # type: ignore
        qo.Rename({"race_source_value": "race"}),  # type: ignore
    )
    persons = SYNTHEA.person()
    persons = persons.ops(ops)
    visits = SYNTHEA.visit_occurrence()
    return visits.join(persons, on="person_id").run()


def measurement_table() -> pd.DataFrame:
    """Get the measurements table."""
    return SYNTHEA.measurement().run()


def get_filtered_dataset(table: pd.DataFrame, filter_func: Callable) -> Dataset:
    """Get filtered dataset.

    Parameters
    ----------
    table : pd.DataFrame
        The dataframe to filter.
    filter_func : Callable
        The filter function to use.

    Returns
    -------
    Dataset
        The filtered dataset.

    """
    pd_ds = Dataset.from_pandas(
        table,
        split=Split.ALL,
        preserve_index=False,
    )
    return pd_ds.filter(
        filter_func,
        batched=True,
        keep_in_memory=True,
        load_from_cache_file=False,
    )


@pytest.mark.integration_test()
def test_overall():
    """Test overall filter."""
    table = visits_table()
    filtered_ds = get_filtered_dataset(table=table, filter_func=overall)
    assert len(table) == len(filtered_ds)


@pytest.mark.integration_test()
@pytest.mark.parametrize("column_name", ["unit_source_value"])
def test_filter_non_null(column_name: str):
    """Test filter non-null."""
    table = measurement_table()
    filtered_ds = get_filtered_dataset(
        table=table,
        filter_func=partial(filter_non_null, column_names=column_name),
    )
    assert None not in filtered_ds.unique(column_name)


@pytest.mark.integration_test()
@pytest.mark.parametrize(
    ("column_name", "value", "negate", "keep_nulls"),
    [
        ("unit_source_value", ["cm", "mm"], False, False),
        ("unit_source_value", "n/a", True, False),
        ("unit_source_value", "n/a", True, True),
    ],
)
def test_filter_value(
    column_name: str,
    value: Any,
    negate: bool,
    keep_nulls: bool,
):
    """Test filter feature value."""
    filter_func = partial(
        filter_value,
        column_name=column_name,
        value=value,
        negate=negate,
        keep_nulls=keep_nulls,
    )
    table = measurement_table()
    filtered_ds = get_filtered_dataset(table=table, filter_func=filter_func)

    if not isinstance(value, list):
        value = [value]

    if value == "n/a" and not keep_nulls:
        value += [None]

    assert np.isin(filtered_ds[column_name], value, invert=negate).all()

    if keep_nulls:
        assert pd.isnull(filtered_ds[column_name]).any()


@pytest.mark.integration_test()
@pytest.mark.parametrize(
    (
        "column_name",
        "min_value",
        "max_value",
        "min_inclusive",
        "max_inclusive",
        "negate",
        "keep_nulls",
    ),
    [
        ("unit_source_value", "cm", "mm", True, True, False, False),
        ("measurement_date", -np.inf, np.inf, True, True, False, False),
        ("measurement_datetime", "2014-01-14", np.inf, True, True, False, False),
        ("value_as_number", 100, 1000, False, False, False, False),
        ("value_as_number", 42, 420, True, True, False, True),
        ("value_as_number", -np.inf, 69, True, False, True, False),
    ],
)
def test_filter_range(
    column_name: str,
    min_value: Any,
    max_value: Any,
    min_inclusive: bool,
    max_inclusive: bool,
    negate: bool,
    keep_nulls: bool,
):
    """Test filter feature value range."""
    filter_func = partial(
        filter_range,
        column_name=column_name,
        min_value=min_value,
        max_value=max_value,
        min_inclusive=min_inclusive,
        max_inclusive=max_inclusive,
        negate=negate,
        keep_nulls=keep_nulls,
    )
    table = measurement_table()

    # Convert min_value and max_value to datetime if necessary.
    min_value, max_value, value_is_datetime = _maybe_convert_to_datetime(
        min_value,
        max_value,
    )

    # If the column is not numeric or datetime, we expect a ValueError.
    col_dtype = (
        pd.Series(
            table[column_name],
            dtype="datetime64[ns]" if value_is_datetime else None,
        )
        .to_numpy()
        .dtype
    )

    if not (
        np.issubdtype(col_dtype, np.number) or np.issubdtype(col_dtype, np.datetime64)
    ):
        with pytest.raises(TypeError):
            filtered_ds = get_filtered_dataset(table=table, filter_func=filter_func)
        return

    # If the min_value is greater than the max_value, we expect a ValueError.
    if min_value > max_value:
        with pytest.raises(ValueError):
            filtered_ds = get_filtered_dataset(table=table, filter_func=filter_func)
        return

    # Otherwise, we expect the filter to work.
    filtered_ds = get_filtered_dataset(table=table, filter_func=filter_func)
    examples = pd.Series(
        filtered_ds[column_name],
        dtype="datetime64[ns]" if value_is_datetime else None,
    ).to_numpy()

    mask = (
        ((examples > min_value) & (examples < max_value))
        | ((examples == min_value) & min_inclusive)
        | ((examples == max_value) & max_inclusive)
    )

    if negate:
        mask = ~mask

    if keep_nulls:
        mask = mask | pd.isnull(examples)

    assert mask.all()


@pytest.mark.integration_test()
@pytest.mark.parametrize(
    ("column_name", "year", "month", "day", "hour", "negate", "keep_nulls"),
    [
        ("visit_start_date", 2014, None, None, None, False, False),
        ("visit_start_date", None, [4, 6, 7], None, None, False, False),
        ("visit_start_date", None, None, [7, 14, 21], None, False, False),
        ("visit_start_date", None, None, None, [0, 12, 23], False, False),
        (
            "visit_start_date",
            [2001, 2007, 2008, 2009],
            [9, 10, 11, 12],
            None,
            None,
            True,
            False,
        ),
        ("visit_start_date", 2014, 1, 14, 0, True, True),
    ],
)
def test_filter_datetime(
    column_name: str,
    year: Union[int, str, List[int], List[str]],
    month: Union[int, List[int]],
    day: Union[int, List[int]],
    hour: Union[int, List[int]],
    negate: bool,
    keep_nulls: bool,
):
    """Test filter feature value datetime."""
    filter_func = partial(
        filter_datetime,
        column_name=column_name,
        year=year,
        month=month,
        day=day,
        hour=hour,
        negate=negate,
        keep_nulls=keep_nulls,
    )
    table = visits_table()

    # add a null row
    table.loc[len(table)] = [None] * len(table.columns)

    filtered_ds = get_filtered_dataset(table=table, filter_func=filter_func)

    examples = pd.to_datetime(filtered_ds[column_name])
    result = np.ones_like(examples, dtype=bool)

    if year is not None:
        result &= np.isin(examples.year, year)

    if month is not None:
        result &= np.isin(examples.month, month)

    if day is not None:
        result &= np.isin(examples.day, day)

    if hour is not None:
        result &= np.isin(examples.hour, hour)

    if negate:
        result = ~result

    if keep_nulls:
        result = np.bitwise_or(result, pd.isnull(examples))
    assert result.all()


@pytest.mark.integration_test()
@pytest.mark.parametrize(
    ("column_name", "contains", "negate", "keep_nulls"),
    [
        ("unit_source_value", "kg", False, False),
        ("unit_source_value", ["kg", "mm"], False, False),
        ("unit_source_value", ["kg", "mm"], True, False),
        ("unit_source_value", ["kg", "mm"], True, True),
    ],
)
def test_filter_string_contains(
    column_name: str,
    contains: str,
    negate: bool,
    keep_nulls: bool,
):
    """Test filter feature value string contains."""
    filter_func = partial(
        filter_string_contains,
        column_name=column_name,
        contains=contains,
        negate=negate,
        keep_nulls=keep_nulls,
    )
    table = measurement_table()

    filtered_ds = get_filtered_dataset(table=table, filter_func=filter_func)
    examples = filtered_ds[column_name]
    col = table[column_name]

    if isinstance(contains, str):
        contains = [contains]

    result = np.zeros_like(col, dtype=bool)
    for substring in contains:
        result |= col.str.contains(substring, case=False).to_numpy(dtype=bool)

    if negate:
        result = ~result

    if keep_nulls:
        result |= pd.isnull(col)
    else:
        result &= pd.notnull(col)

    expected = table[result][column_name]
    assert np.array_equal(examples, expected)


@pytest.mark.integration_test()
@pytest.mark.parametrize(
    (
        "value_col",
        "value",
        "range_col",
        "range_min",
        "range_max",
        "datetime_col",
        "year",
        "month",
    ),
    [
        (
            "race",
            ["asian", "black"],
            "visit_start_date",
            "2014-01-01",
            "2014-12-31",
            "visit_end_datetime",
            [2014, 2015, 2016, 2017, 2018],
            [1, 2, 3, 4, 5, 10, 11, 12],
        ),
    ],
)
def test_compound_filter(
    value_col: str,
    value: Any,
    range_col: str,
    range_min: Any,
    range_max: Any,
    datetime_col: str,
    year: Union[int, str, List[int], List[str]],
    month: Union[int, List[int]],
):
    """Test compound feature value filter."""
    # filter specific value(s)
    filter_func_values = partial(
        filter_value,
        column_name=value_col,
        value=value,
        negate=False,
        keep_nulls=False,
    )
    table = visits_table()

    # filter range
    filter_func_range = partial(
        filter_range,
        column_name=range_col,
        min_value=range_min,
        max_value=range_max,
        min_inclusive=False,
        max_inclusive=False,
        negate=False,
        keep_nulls=False,
    )

    # filter datetime
    filter_func_datetime = partial(
        filter_datetime,
        column_name=datetime_col,
        year=year,
        month=month,
        day=None,
        hour=None,
        negate=False,
        keep_nulls=False,
    )

    filtered_ds = get_filtered_dataset(
        table=table,
        filter_func=partial(
            compound_filter,
            slice_functions=[
                filter_func_values,
                filter_func_range,
                filter_func_datetime,
            ],
        ),
    )

    range_min, range_max, value_is_datetime = _maybe_convert_to_datetime(
        range_min,
        range_max,
    )
    range_examples = pd.Series(
        filtered_ds[range_col],
        dtype="datetime64[ns]" if value_is_datetime else None,
    ).to_numpy()

    # check that the filtered dataset has the correct values
    assert np.all(
        np.isin(filtered_ds[value_col], value)
        & (
            (np.greater(range_examples, range_min))
            & (np.less(range_examples, range_max))
        )
        & (
            np.isin(
                pd.to_datetime(filtered_ds[datetime_col]).year,
                year,
            )
            & np.isin(
                pd.to_datetime(filtered_ds[datetime_col]).month,
                month,
            )
        ),
    )


@pytest.mark.integration_test()
def test_slice_spec():
    """Test SliceSpec class."""
    value1 = ["mmHg", "kg", "mL", "mL/min"]
    value2 = "cm"
    min_value = 100
    max_value = 1000
    year = [2000, 2005, 2010, 2015]
    month = [1, 2, 3, 4, 5, 10, 11, 12]

    spec_list = [
        {"unit_source_value": {"keep_nulls": False}},
        {"value_as_number": {"keep_nulls": False}},
        {"unit_source_value": {"value": value1}},
        {"value_as_number": {"min_value": min_value, "max_value": max_value}},
        {"measurement_datetime": {"year": year}},
        {
            "unit_source_value": {"value": value2},
            "value_as_number": {
                "max_value": max_value,
                "negate": True,
                "keep_nulls": False,
            },
            "measurement_datetime": {"month": month},
        },
    ]

    table = measurement_table()

    slice_spec = SliceSpec(spec_list=spec_list, column_names=table.columns)
    assert slice_spec.spec_list == spec_list

    for slice_name, slice_func in slice_spec.slices():
        assert callable(slice_func)

        filtered_ds = get_filtered_dataset(table, filter_func=slice_func)

        # check that the filtered dataset has the correct values
        if slice_name == "unit_source_value:non_null":
            assert None not in filtered_ds.unique("unit_source_value")
        if slice_name == "value_as_number:non_null":
            assert None not in filtered_ds.unique("value_as_number")
        if "unit_source_value:['mmHg', 'kg', 'mL', 'mL/min']" in slice_name:
            assert np.all(
                np.isin(
                    filtered_ds["unit_source_value"],
                    ["mmHg", "kg", "mL", "mL/min"],
                ),
            )
        if f"value_as_number:[{min_value} - {max_value}]" in slice_name:
            assert np.all(
                np.greater_equal(filtered_ds["value_as_number"], min_value)
                & np.less_equal(filtered_ds["value_as_number"], max_value),
            )
        if f"measurement_datetime:year={year}" in slice_name:
            assert np.all(
                np.isin(
                    pd.to_datetime(filtered_ds["measurement_datetime"]).year,
                    year,
                ),
            )
        if (
            f"unit_source_value:cm&!(value_as_number:-inf - {max_value})&measurement_datetime:month={month}"  # noqa: E501
            in slice_name
        ):
            assert np.all(
                np.isin(filtered_ds["unit_source_value"], value2)
                & (np.greater(filtered_ds["value_as_number"], max_value))
                & np.isin(
                    pd.to_datetime(filtered_ds["measurement_datetime"]).month,
                    month,
                ),
            )
