"""Model implementations."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from models.catalog import register_model
from models.neural_nets.gru import GRUModel
from models.neural_nets.lstm import LSTMModel
from models.neural_nets.mlp import MLPModel
from models.neural_nets.rnn import RNNModel

register_model("rf", model_type="static")(RandomForestClassifier)
register_model("xgb", model_type="static")(XGBClassifier)
register_model("lr", model_type="static")(LogisticRegression)
register_model("mlp", model_type="static")(MLPClassifier)
