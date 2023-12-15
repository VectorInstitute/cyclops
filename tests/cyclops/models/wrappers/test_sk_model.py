"""Tests for scikit-learn model wrapper."""

import pandas as pd
from datasets import Dataset
from sklearn.datasets import load_diabetes

from cyclops.models import create_model
from cyclops.models.wrappers import SKModel


def test_find_best_grid_search():
    """Test find_best method with grid search."""
    parameters = {"C": [1], "l1_ratio": [0.5]}
    X, y = load_diabetes(return_X_y=True)
    metric = "accuracy"
    method = "grid"

    model = create_model("logistic_regression", penalty="elasticnet", solver="saga")
    best_estimator = model.find_best(
        parameters=parameters,
        X=X,
        y=y,
        metric=metric,
        method=method,
    )
    assert isinstance(best_estimator, SKModel)


def test_find_best_random_search():
    """Test find_best method with random search."""
    parameters = {"alpha": [0.001], "hidden_layer_sizes": [10]}
    X, y = load_diabetes(return_X_y=True)
    metric = "accuracy"
    method = "random"

    model = create_model("mlp_classifier", early_stopping=True)
    best_estimator = model.find_best(
        parameters=parameters,
        X=X,
        y=y,
        metric=metric,
        method=method,
    )
    assert isinstance(best_estimator, SKModel)


def test_find_best_hf_dataset_input():
    """Test find_best method with huggingface dataset input."""
    parameters = {"alpha": [0.001], "hidden_layer_sizes": [10]}
    data = load_diabetes(as_frame=True)
    X, y = data["data"], data["target"]
    X_y = pd.concat([X, y], axis=1)
    features_names = data["feature_names"]
    dataset = Dataset.from_pandas(X_y)
    metric = "accuracy"
    method = "random"

    model = create_model("mlp_classifier", early_stopping=True)
    best_estimator = model.find_best(
        parameters=parameters,
        X=dataset,
        metric=metric,
        method=method,
        feature_columns=features_names,
        target_columns="target",
    )
    assert isinstance(best_estimator, SKModel)
