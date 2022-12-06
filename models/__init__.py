"""Model implementations."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from models.catalog import register_model

register_model("random_forest", model_type="static")(RandomForestClassifier)
register_model("xgb_classifier", model_type="static")(XGBClassifier)
register_model("logistic_regression", model_type="static")(LogisticRegression)
register_model("mlp", model_type="static")(MLPClassifier)
