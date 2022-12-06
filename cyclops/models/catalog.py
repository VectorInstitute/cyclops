"""Model catalog."""

from typing import Optional, TypeVar, Union

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as MLP
from xgboost import XGBClassifier

from cyclops.models.neural_nets.gru import GRUModel
from cyclops.models.neural_nets.lstm import LSTMModel
from cyclops.models.neural_nets.mlp import MLPModel
from cyclops.models.neural_nets.rnn import RNNModel

MODELS = {
    "gru": GRUModel,
    "lstm": LSTMModel,
    "rnn": RNNModel,
    "mlp_pt": MLPModel,
    "lr": LR,
    "mlp": MLP,
    "rf": RF,
    "xgb": XGBClassifier,
}

_PTModel = TypeVar("_PTModel", GRUModel, LSTMModel, RNNModel, MLPModel)
_SKModel = TypeVar("_SKModel", LR, MLP, RF, XGBClassifier)
_Model = Union[_PTModel, _SKModel]

STATIC_MODELS = ["lr", "mlp", "mlp_pt", "rf", "xgb"]
TEMPORAL_MODELS = ["gru", "rnn", "lstm"]

PT_MODELS = ["gru", "mlp_pt", "rnn", "lstm"]
SK_MODELS = ["lr", "mlp", "rf", "xgb"]

MODEL_CATALOG = {}


def register_model(model: type, name: Optional[str] = None):
    """Register model with dict.

    Parameters
    ----------
    model: type
        Model implementation wrapped in a class.
    name: str, optional

    """
    if not name:
        name = model.__name__
    if name not in MODEL_CATALOG:
        MODEL_CATALOG[name] = model


def get_model(name: str) -> Optional[type]:
    """Get model from catalog.

    Parameters
    ----------
    name: str
        Model name.

    Returns
    -------
    type
        Model class.

    Raises
    ------
    NotImplementedError
        If model name provided is not in the catalog.

    """
    if name not in MODEL_CATALOG:
        raise NotImplementedError
    return MODEL_CATALOG.get(name)
