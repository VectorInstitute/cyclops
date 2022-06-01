import numpy as np
import pandas as pd
import torch
from dataset import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from temporal_models import GRUModel, LSTMModel, RNNModel


def pandas_to_dataset(
    X: pd.DataFrame,
    y: list,
):
    """Convert pandas dataframe to dataset.

    Parameters
    ----------
    X: pandas.DataFrame
        Dataset as a pandas dataframe.
    y: list
        List of labels.

    """
    X = X.unstack(fill_value=np.nan, level=0).stack().swaplevel(0, 1)
    samples = len(X.index.unique(level=0))
    timesteps = len(X.index.unique(level=1))
    features = X.shape[1]
    inputs = torch.tensor(
        X.values.reshape(samples, timesteps, features), dtype=torch.float32
    )
    target = torch.tensor(y.reshape(len(y), 1), dtype=torch.float32)
    return Data(inputs, target)


def feature_label_split(data, target_col):
    """Split dataset into features and label.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataset containing features and labels.
    target_col: string
        Target column for prediction.

    """
    y = data[[target_col]]
    X = data.drop(columns=[target_col])
    return X, y


def train_val_test_split(data, target_col, test_ratio):
    """Split dataset into train, validation and test set.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Dataset containing features and labels.
    target_col: String
        Target column for prediction.
    test_ratio: float
        Proportion of dataset to include in the test split.

    """
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_scaler(scaler):
    """Get scaler.

    Parameters
    ----------
    scaler: string
        String indicating which scaler to retrieve.

    """
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


def get_temporal_model(model, model_params):
    """Get temporal model.

    Parameters
    ----------
    model: string
        String with model name (e.g. rnn, lstm, gru).

    """
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)
