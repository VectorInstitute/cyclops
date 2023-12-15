"""Tests for scikit-learn model wrapper."""

import numpy as np

from cyclops.models import create_model


def test_find_best_grid_search():
    """Test find_best method with grid search."""
    parameters = {"C": [1, 2, 3], "l1_ratio": [0.25, 0.5, 0.75]}
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [1, 3, 1],
            [2, 3, 2],
            [3, 3, 3],
            [1, 2, 1],
            [2, 2, 1],
            [3, 2, 1],
            [1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
        ],
    )
    y = np.array([1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2])
    feature_columns = ["feature1", "feature2", "feature3"]
    target_columns = ["target"]
    transforms = None
    metric = "accuracy"
    method = "grid"

    model = create_model("logistic_regression")
    best_estimator = model.find_best(
        parameters,
        X,
        y,
        feature_columns,
        target_columns,
        transforms,
        metric,
        method,
    )
    assert best_estimator.l1_ratio == 0.25
    assert best_estimator.C == 1
