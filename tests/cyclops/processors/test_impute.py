"""Test imputation functions."""


import numpy as np
import pandas as pd
import pytest

from cyclops.processors.column_names import ENCOUNTER_ID, TIMESTEP
from cyclops.processors.constants import MEAN, MEDIAN
from cyclops.processors.impute import (
    Imputer,
    impute_features,
    remove_encounters_if_missing,
    remove_features_if_missing,
)


@pytest.fixture
def test_input_feature_static():
    """Create test input static features."""
    input_ = pd.DataFrame(
        index=[0, 1, 2, 4],
        columns=["A", "B", "C"],
    )
    input_.loc[0] = [1, 10, 2.3]
    input_.loc[1] = [np.nan, np.nan, 1.2]
    input_.loc[2] = [1, 1.09, 3.2]
    input_.loc[4] = [0, np.nan, 2.3]
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


def test_remove_encounters_if_missing(  # pylint: disable=redefined-outer-name
    test_input_feature_static, test_input_feature_temporal
):
    """Test remove_encounters_if_missing fn."""
    features = remove_encounters_if_missing(test_input_feature_static, 0.25)
    # assert 1 not in features.index

    features = remove_encounters_if_missing(test_input_feature_temporal, 0.25)
    # assert 0 not in features.index.get_level_values(0)
    print(features)


def test_remove_features_if_missing(  # pylint: disable=redefined-outer-name
    test_input_feature_static, test_input_feature_temporal
):
    """Test remove_features_if_missing fn."""
    features = remove_features_if_missing(test_input_feature_static, 0.25)
    assert "B" not in features
    features = remove_features_if_missing(test_input_feature_temporal, 0.4)
    assert "B" not in features and "A" not in features


def test_impute_features(  # pylint: disable=redefined-outer-name
    test_input_feature_static,
):
    """Test impute_features fn."""
    imputer = Imputer()
    features = impute_features(test_input_feature_static, imputer=imputer)
    assert features.equals(test_input_feature_static)

    imputer = Imputer(strategy=MEAN)
    features = impute_features(test_input_feature_static, imputer=imputer)

    assert features["A"][1] == features["A"].mean(skipna=True)
    assert features["B"][1] == features["B"].mean(skipna=True)
    assert features["B"][4] == features["B"].mean(skipna=True)

    imputer = Imputer(strategy=MEDIAN)
    features = impute_features(test_input_feature_static, imputer=imputer)

    assert features["A"][1] == features["A"].median(skipna=True)
    assert features["B"][1] == features["B"].median(skipna=True)
    assert features["B"][4] == features["B"].median(skipna=True)
