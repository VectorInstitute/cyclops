"""Model implementations."""

from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier

from cyclops.models.catalog import create_model, list_models, register_model, wrap_model
from cyclops.models.neural_nets import GRUModel, LSTMModel, MLPModel, RNNModel
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torchxraysion.models
    import xgboost
else:
    DenseNet = import_optional_module(
        "torchxrayvision.models",
        attribute="DenseNet",
        error="warn",
    )
    ResNet = import_optional_module(
        "torchxrayvision.models",
        attribute="ResNet",
        error="warn",
    )
    XGBClassifier = import_optional_module(
        "xgboost",
        attribute="XGBClassifier",
        error="warn",
    )


register_model(name="sgd_classifier", model_type="static")(SGDClassifier)
register_model(name="sgd_regressor", model_type="static")(SGDRegressor)
register_model("rf_classifier", model_type="static")(RandomForestClassifier)
register_model("logistic_regression", model_type="static")(LogisticRegression)
register_model("mlp", model_type="static")(MLPClassifier)
if XGBClassifier is not None:
    register_model("xgb_classifier", model_type="static")(XGBClassifier)
if DenseNet is not None:
    register_model("densenet", model_type="image")(DenseNet)
if ResNet is not None:
    register_model("resnet", model_type="image")(ResNet)
if GRUModel is not None:
    register_model("gru", model_type="temporal")(GRUModel)
if LSTMModel is not None:
    register_model("lstm", model_type="temporal")(LSTMModel)
if MLPModel is not None:
    register_model("mlp", model_type="static")(MLPModel)
if RNNModel is not None:
    register_model("rnn", model_type="temporal")(RNNModel)
