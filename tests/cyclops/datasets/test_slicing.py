"""Tests for cyclops.evaluate.slicing module."""
from functools import partial
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset
from datasets.splits import Split

import cyclops.query.ops as qo
from cyclops.datasets.slicing import (
    _maybe_convert_to_datetime,  # pylint: disable=protected-access
)
from cyclops.datasets.slicing import (
    SlicingConfig,
    filter_compound_feature_value,
    filter_feature_value,
    filter_feature_value_datetime,
    filter_feature_value_range,
    filter_non_null,
    no_filter,
)
from cyclops.query.omop import OMOPQuerier

SYNTHEA = OMOPQuerier("cdm_synthea10", database="synthea_integration_test")


@pytest.fixture
def visits_table() -> pd.DataFrame:
    """Get the visits table."""
    ops = qo.Sequential(
        [
            qo.ConditionEquals("gender_source_value", "M"),  # type: ignore
            qo.Rename({"race_source_value": "race"}),  # type: ignore
        ]
    )

    persons_qi = SYNTHEA.person(ops=ops)
    visit_occurrence_table = SYNTHEA.visit_occurrence(
        join=qo.JoinArgs(join_table=persons_qi.query, on="person_id")
    ).run()

    return visit_occurrence_table


@pytest.fixture
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
    filtered_ds = pd_ds.filter(
        filter_func,
        batched=True,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
    return filtered_ds


@pytest.mark.integration_test
def test_no_filter(visits_table: pd.DataFrame):
    """Test no filter."""
    filtered_ds = get_filtered_dataset(visits_table, no_filter)
    assert len(visits_table) == len(filtered_ds)


@pytest.mark.integration_test
@pytest.mark.parametrize("column_name", ["unit_source_value"])
def test_filter_non_null(column_name: str, measurement_table: pd.DataFrame):
    """Test filter non-null."""
    filtered_ds = get_filtered_dataset(
        table=measurement_table,
        filter_func=partial(filter_non_null, feature_keys=column_name),
    )
    assert None not in filtered_ds.unique(column_name)


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "column_name,value,negate, keep_nulls",
    [
        ("unit_source_value", ["cm", "mm"], False, False),
        ("unit_source_value", "n/a", True, False),
        ("unit_source_value", "n/a", True, True),
    ],
)
def test_filter_feature_value(
    measurement_table: pd.DataFrame,
    column_name: str,
    value: Any,
    negate: bool,
    keep_nulls: bool,
):
    """Test filter feature value."""
    filter_func = partial(
        filter_feature_value,
        feature_key=column_name,
        value=value,
        negate=negate,
        keep_nulls=keep_nulls,
    )
    filtered_ds = get_filtered_dataset(table=measurement_table, filter_func=filter_func)

    if not isinstance(value, list):
        value = [value]

    if value == "n/a" and not keep_nulls:
        value += [None]

    assert np.isin(filtered_ds[column_name], value, invert=negate).all()

    if keep_nulls:
        assert pd.isnull(filtered_ds[column_name]).any()


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "column_name,min_value,max_value,min_inclusive,max_inclusive,negate,keep_nulls",
    [
        ("unit_source_value", "cm", "mm", True, True, False, False),
        ("measurement_date", -np.inf, np.inf, True, True, False, False),
        ("measurement_datetime", "2014-01-14", np.inf, True, True, False, False),
        ("value_as_number", 100, 1000, False, False, False, False),
        ("value_as_number", 42, 420, True, True, False, True),
        ("value_as_number", -np.inf, 69, True, False, True, False),
    ],
)
def test_filter_feature_value_range(
    measurement_table: pd.DataFrame,
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
        filter_feature_value_range,
        feature_key=column_name,
        min_value=min_value,
        max_value=max_value,
        min_inclusive=min_inclusive,
        max_inclusive=max_inclusive,
        negate=negate,
        keep_nulls=keep_nulls,
    )

    # Convert min_value and max_value to datetime if necessary.
    min_value, max_value, value_is_datetime = _maybe_convert_to_datetime(
        min_value, max_value
    )

    # If the column is not numeric or datetime, we expect a ValueError.
    col_dtype = (
        pd.Series(
            measurement_table[column_name],
            dtype="datetime64[ns]" if value_is_datetime else None,
        )
        .to_numpy()
        .dtype
    )

    if not (
        np.issubdtype(col_dtype, np.number) or np.issubdtype(col_dtype, np.datetime64)
    ):
        with pytest.raises(ValueError):
            filtered_ds = get_filtered_dataset(
                table=measurement_table, filter_func=filter_func
            )
        return

    # If the min_value is greater than the max_value, we expect a ValueError.
    if min_value > max_value:
        with pytest.raises(ValueError):
            filtered_ds = get_filtered_dataset(
                table=measurement_table, filter_func=filter_func
            )
        return

    # Otherwise, we expect the filter to work.
    filtered_ds = get_filtered_dataset(table=measurement_table, filter_func=filter_func)
    examples = pd.Series(
        filtered_ds[column_name], dtype="datetime64[ns]" if value_is_datetime else None
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


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "column_name,year,month,day,hour,negate,keep_nulls",
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
def test_filter_feature_value_datetime(
    visits_table: pd.DataFrame,
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
        filter_feature_value_datetime,
        feature_key=column_name,
        year=year,
        month=month,
        day=day,
        hour=hour,
        negate=negate,
        keep_nulls=keep_nulls,
    )

    # add a null row
    visits_table.loc[len(visits_table)] = [None] * len(visits_table.columns)

    filtered_ds = get_filtered_dataset(table=visits_table, filter_func=filter_func)

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


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "value_col,value,range_col,range_min,range_max, datetime_col,year,month",
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
        )
    ],
)
def test_compound_feature_value(
    visits_table: pd.DataFrame,
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
        filter_feature_value,
        feature_key=value_col,
        value=value,
        negate=False,
        keep_nulls=False,
    )

    # filter range
    filter_func_range = partial(
        filter_feature_value_range,
        feature_key=range_col,
        min_value=range_min,
        max_value=range_max,
        min_inclusive=False,
        max_inclusive=False,
        negate=False,
        keep_nulls=False,
    )

    # filter datetime
    filter_func_datetime = partial(
        filter_feature_value_datetime,
        feature_key=datetime_col,
        year=year,
        month=month,
        day=None,
        hour=None,
        negate=False,
        keep_nulls=False,
    )

    filtered_ds = get_filtered_dataset(
        table=visits_table,
        filter_func=partial(
            filter_compound_feature_value,
            slice_functions=[
                filter_func_values,
                filter_func_range,
                filter_func_datetime,
            ],
        ),
    )

    range_min, range_max, value_is_datetime = _maybe_convert_to_datetime(
        range_min, range_max
    )
    range_examples = pd.Series(
        filtered_ds[range_col], dtype="datetime64[ns]" if value_is_datetime else None
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
        )
    )


# test SlcingConfig
def test_slicing_config(measurement_table: pd.DataFrame):
    """Test SlicingConfig class."""
    value1 = ["mmHg", "kg", "mL", "mL/min"]
    value2 = ["mm", "cm"]
    min_value = 100
    max_value = 1000
    year = [2000, 2005, 2010, 2015]
    month = [1, 2, 3, 4, 5, 10, 11, 12]

    feature_keys = ["unit_source_value", "value_as_number"]
    feature_values = [
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

    slice_config = SlicingConfig(
        feature_keys=feature_keys,
        feature_values=feature_values,
        column_names=measurement_table.columns,
    )
    assert slice_config.feature_keys == feature_keys
    assert slice_config.feature_values == feature_values

    for slice_name, slice_func in slice_config.get_slices().items():
        assert callable(slice_func)

        filtered_ds = get_filtered_dataset(measurement_table, filter_func=slice_func)

        # check that the filtered dataset has the correct values
        if slice_name == "filter_non_null:unit_source_value":
            assert None not in filtered_ds.unique("unit_source_value")
        if slice_name == "filter_non_null:value_as_number":
            assert None not in filtered_ds.unique("value_as_number")
        if "filter_feature_value" in slice_name:
            assert np.all(
                np.isin(
                    filtered_ds["unit_source_value"], ["mmHg", "kg", "mL", "mL/min"]
                )
            )
        if "filter_feature_value_range" in slice_name:
            assert np.all(
                np.greater(filtered_ds["value_as_number"], min_value)
                & np.less(filtered_ds["value_as_number"], max_value)
            )
        if "filter_feature_value_datetime" in slice_name:
            assert np.all(
                np.isin(
                    pd.to_datetime(filtered_ds["measurement_datetime"]).year,
                    year,
                )
            )
        if "filter_compound_feature_value" in slice_name:
            assert np.all(
                np.isin(filtered_ds["unit_source_value"], value2)
                & (np.greater(filtered_ds["value_as_number"], max_value))
                & np.isin(
                    pd.to_datetime(filtered_ds["measurement_datetime"]).month,
                    month,
                )
            )
