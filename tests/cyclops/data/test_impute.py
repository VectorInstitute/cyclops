"""Test imputation functions."""

import numpy as np
import pandas as pd
import pytest

from cyclops.data.constants import (
    BFILL,
    DROP,
    EXTRA,
    FFILL,
    FFILL_BFILL,
    IGNORE,
    INTER,
    LINEAR_INTERP,
    MEAN,
    MEDIAN,
    MODE,
)
from cyclops.data.impute import (
    SeriesImputer,
    TabularImputer,
    compute_inter_range,
    np_bfill,
    np_ffill,
    np_ffill_bfill,
    np_fill_null_mean,
    np_fill_null_num,
    np_fill_null_zero,
)


ENCOUNTER_ID = "enc_id"
TIMESTEP = "timestep"


@pytest.fixture()
def test_tabular():
    """Create test input tabular features."""
    return pd.DataFrame(
        data=[
            [np.nan, 10, 2.3],
            [1, np.nan, 1.2],
            [np.nan, 1.09, np.nan],
            [2, np.nan, 2.3],
            [np.nan, 1.09, np.nan],
        ],
        columns=["A", "B", "C"],
    )


@pytest.fixture()
def test_input_feature_temporal():
    """Create test input temporal features."""
    index = pd.MultiIndex.from_product(
        [[0, 2], range(6)],
        names=[ENCOUNTER_ID, TIMESTEP],
    )
    input_ = pd.DataFrame(index=index, columns=["A", "B", "C"])
    input_.loc[(0, 0)] = [np.nan, np.nan, 1]
    input_.loc[(0, 1)] = [1, 1.09, 2]
    input_.loc[(0, 2)] = [0, np.nan, 2.3]
    input_.loc[(2, 0)] = [0, 2.1, 2.3]
    input_.loc[(2, 1)] = [0, 9.1, 0.3]
    input_.loc[(2, 2)] = [0, 2.1, 1.3]
    input_.loc[(2, 3)] = [1, 5.8, 0.3]
    input_.loc[(2, 4)] = [1, 2.0, 1.3]
    return input_


def test_np_ffill():
    """Test np_ffill."""
    test_arr = np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6])
    res = np_ffill(test_arr)
    assert np.array_equal(
        res,
        np.array([np.nan, 1, 2, 2, 3, 4, 4, 5, 6]),
        equal_nan=True,
    )


def test_np_bfill():
    """Test np_bfill."""
    test_arr = np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6])
    res = np_bfill(test_arr)
    assert np.array_equal(res, np.array([1, 1, 2, 3, 3, 4, 5, 5, 6]), equal_nan=True)


def test_np_ffill_bfill():
    """Test np_ffill_bfill."""
    test_arr = np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6])
    res = np_ffill_bfill(test_arr)
    assert np.array_equal(res, np.array([1, 1, 2, 2, 3, 4, 4, 5, 6]), equal_nan=True)

    test_arr = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
    res = np_ffill_bfill(test_arr)
    assert np.array_equal(
        res,
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        equal_nan=True,
    )

    test_arr = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7])
    res = np_ffill_bfill(test_arr)
    assert np.array_equal(res, np.array([1, 1, 2, 3, 4, 5, 6, 7]), equal_nan=True)

    test_arr = np.array([1, 2, 3, 4, 5, 6, 7, np.nan])
    res = np_ffill_bfill(test_arr)
    assert np.array_equal(res, np.array([1, 2, 3, 4, 5, 6, 7, 7]), equal_nan=True)


def test_np_fill_null_num():
    """Test np_fill_null_num."""
    test_arr = np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6])
    res = np_fill_null_num(test_arr, 0)
    assert np.array_equal(res, np.array([0, 1, 2, 0, 3, 4, 0, 5, 6]))

    test_arr = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
    res = np_fill_null_num(test_arr, 0)
    assert np.array_equal(res, np.array([0, 0, 0, 0, 0, 0, 0, 0]))


def test_np_fill_null_zero():
    """Test np_fill_null_zero."""
    test_arr = np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6])
    res = np_fill_null_zero(test_arr)
    assert np.array_equal(res, np.array([0, 1, 2, 0, 3, 4, 0, 5, 6]))

    test_arr = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
    res = np_fill_null_zero(test_arr)
    assert np.array_equal(res, np.array([0, 0, 0, 0, 0, 0, 0, 0]))


def test_np_fill_null_mean():
    """Test np_fill_null_mean."""
    test_arr = np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6])
    res = np_fill_null_mean(test_arr)
    assert np.array_equal(res, np.array([3.5, 1, 2, 3.5, 3, 4, 3.5, 5, 6]))


def test_compute_inter_range():
    """Test the compute_inter_range function."""
    null = pd.Series([True, True, False, True, False, True, True, True])
    inter_range = compute_inter_range(null)
    inter_start, inter_stop = inter_range
    assert np.array_equal(null[inter_start:inter_stop], np.array([False, True, False]))

    null = pd.Series([True, True, True, True, True, True, True, True])
    inter_range = compute_inter_range(null)
    assert inter_range is None


def test_series_imputation():
    """Test the SeriesImputer."""
    series = pd.Series([np.nan, np.nan, 1.0, np.nan, 1.0, 4.0, np.nan, np.nan])

    # No limit
    imputer = SeriesImputer(MEDIAN)
    res, missingness = imputer(series.copy())
    assert missingness == 5 / 8
    assert res[0] == 1.0
    assert res[1] == 1.0
    assert res[3] == 1.0
    assert res[6] == 1.0
    assert res[7] == 1.0

    # INTER
    imputer = SeriesImputer(MEAN, limit_area=INTER)
    res, _ = imputer(series.copy())
    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert res[3] == 2.0
    assert np.isnan(res[6])
    assert np.isnan(res[7])

    res[series.isna()] = np.nan
    assert res.equals(series)

    # EXTRA
    imputer = SeriesImputer(
        lambda series, _: series.fillna(100),
        limit_area=EXTRA,
    )
    res, _ = imputer(series.copy())
    assert res[0] == 100
    assert res[1] == 100
    assert np.isnan(res[3])
    assert res[6] == 100
    assert res[7] == 100

    res[series.isna()] = np.nan
    assert res.equals(series)


def test_tabular_imputation(
    test_tabular,
):
    """Test the TabularImputer and SeriesImputer."""
    imputer = TabularImputer(
        {
            "A": SeriesImputer(IGNORE),
            "B": SeriesImputer(MEAN),
            "C": SeriesImputer(MEDIAN),
        },
    )
    res, missingness = imputer(test_tabular.copy())
    assert missingness["A"] == 3 / 5
    assert missingness["B"] == 2 / 5
    assert missingness["C"] == 2 / 5

    assert np.isnan(res.iloc[0]["A"])
    assert np.isnan(res.iloc[2]["A"])
    assert np.isnan(res.iloc[4]["A"])

    assert res.iloc[1]["B"] == test_tabular["B"].mean()
    assert res.iloc[3]["B"] == test_tabular["B"].mean()

    assert res.iloc[2]["C"] == test_tabular["C"].median()
    assert res.iloc[4]["C"] == test_tabular["C"].median()

    res[test_tabular.isna()] = np.nan
    assert res.equals(test_tabular)

    imputer = TabularImputer(
        {
            "A": SeriesImputer(FFILL_BFILL),
            "B": SeriesImputer(FFILL),
            "C": SeriesImputer(BFILL),
        },
    )
    res, _ = imputer(test_tabular.copy())

    assert res.iloc[0]["A"] == 1.0
    assert res.iloc[2]["A"] == 1.0
    assert res.iloc[4]["A"] == 2.0
    assert res.iloc[1]["B"] == 10.0
    assert res.iloc[3]["B"] == 1.09
    assert res.iloc[2]["C"] == 2.3
    assert np.isnan(res.iloc[4]["C"])

    res[test_tabular.isna()] = np.nan
    assert res.equals(test_tabular)

    imputer = TabularImputer(
        {
            "A": SeriesImputer(FFILL),
        },
    )
    res, _ = imputer(test_tabular.copy())
    assert np.isnan(res.iloc[0]["A"])

    imputer = TabularImputer(
        {
            "B": SeriesImputer(MODE),
            "C": SeriesImputer(LINEAR_INTERP),
        },
    )
    res, _ = imputer(test_tabular.copy())

    assert res.iloc[1]["B"] == 1.09
    assert res.iloc[3]["B"] == 1.09
    assert res.iloc[2]["C"] == (res.iloc[1]["C"] + res.iloc[3]["C"]) / 2
    assert res.iloc[4]["C"] == res.iloc[3]["C"]

    # Test DROP strategy error
    try:
        TabularImputer(
            {
                "A": SeriesImputer(DROP),
            },
        )
        raise ValueError("Should have raised an error. Can't use DROP strategy here.")
    except ValueError:
        pass

    # Test not allowing nulls
    imputer = TabularImputer(
        {
            "A": SeriesImputer(FFILL, allow_nulls_returned=False),
        },
    )

    try:
        imputer(test_tabular.copy())
        raise ValueError(
            "Should have raised an error. FFILL will return nulls, which was unallowed.",
        )
    except ValueError:
        pass
