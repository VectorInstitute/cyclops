"""Test imputation functions."""


import numpy as np
import pandas as pd
import pytest

from cyclops.processors.column_names import ENCOUNTER_ID, TIMESTEP
from cyclops.processors.constants import (
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
from cyclops.processors.impute import SeriesImputer, TabularImputer, compute_inter_range


@pytest.fixture
def test_tabular():
    """Create test input tabular features."""
    input_ = pd.DataFrame(
        data=[
            [np.nan, 10, 2.3],
            [1, np.nan, 1.2],
            [np.nan, 1.09, np.nan],
            [2, np.nan, 2.3],
            [np.nan, 1.09, np.nan],
        ],
        columns=["A", "B", "C"],
    )
    return input_


@pytest.fixture
def test_input_feature_temporal():
    """Create test input temporal features."""
    index = pd.MultiIndex.from_product(
        [[0, 2], range(6)], names=[ENCOUNTER_ID, TIMESTEP]
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


def test_tabular_imputation(  # pylint: disable=redefined-outer-name
    test_tabular,
):
    """Test the TabularImputer and SeriesImputer."""
    # def non_nulls_same(original: pd.DataFrame, imputed: pd.DataFrame) -> bool:

    # print("\n\nINPUT")
    # print(test_tabular)
    # Test various imputation strategies
    imputer = TabularImputer(
        {
            "A": SeriesImputer(IGNORE),
            "B": SeriesImputer(MEAN),
            "C": SeriesImputer(MEDIAN),
        }
    )
    res, missingness = imputer(test_tabular.copy())
    assert missingness["A"] == 3 / 5
    assert missingness["B"] == 2 / 5
    assert missingness["C"] == 2 / 5

    # print("\n\nIGNORE, MEAN, MEDIAN")
    # print(res)

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
        }
    )
    res, _ = imputer(test_tabular.copy())
    # print("\n\nFFILL_BFILL, FFILL, BFILL")
    # print(res)

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
        }
    )
    res, _ = imputer(test_tabular.copy())
    assert np.isnan(res.iloc[0]["A"])

    imputer = TabularImputer(
        {
            "B": SeriesImputer(MODE),
            "C": SeriesImputer(LINEAR_INTERP),
        }
    )
    res, _ = imputer(test_tabular.copy())
    # print("\n\nMODE, LINEAR_INTERP")
    # print(res)

    assert res.iloc[1]["B"] == 1.09
    assert res.iloc[3]["B"] == 1.09
    assert res.iloc[2]["C"] == (res.iloc[1]["C"] + res.iloc[3]["C"]) / 2
    assert res.iloc[4]["C"] == res.iloc[3]["C"]

    # Test DROP strategy error
    try:
        TabularImputer(
            {
                "A": SeriesImputer(DROP),
            }
        )
        raise ValueError("Should have raised an error. Can't use DROP strategy here.")
    except ValueError:
        pass

    # Test not allowing nulls
    imputer = TabularImputer(
        {
            "A": SeriesImputer(FFILL, allow_nulls_returned=False),
        }
    )

    try:
        imputer(test_tabular.copy())
        raise ValueError(
            "Should have raised an error. FFILL will return nulls, which was unallowed."
        )
    except ValueError:
        pass
