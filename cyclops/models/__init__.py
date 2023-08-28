"""Model implementations."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier
from torchxrayvision.models import DenseNet, ResNet
from xgboost import XGBClassifier

from cyclops.models.catalog import create_model, list_models, register_model, wrap_model
from cyclops.models.neural_nets import GRUModel, LSTMModel, MLPModel, RNNModel


register_model(name="sgd_classifier", model_type="static")(SGDClassifier)
register_model(name="sgd_regressor", model_type="static")(SGDRegressor)
register_model("rf_classifier", model_type="static")(RandomForestClassifier)
register_model("xgb_classifier", model_type="static")(XGBClassifier)
register_model("logistic_regression", model_type="static")(LogisticRegression)
register_model("mlp", model_type="static")(MLPClassifier)
register_model("densenet", model_type="image")(DenseNet)
register_model("resnet", model_type="image")(ResNet)
