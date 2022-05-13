from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from data import Data
from rnn import RNNModel
from lstm import LSTMModel
from gru import GRUModel

def pandas_to_dataset(
    X: pd.DataFrame,
    y: list,
):
    """Convert pandas dataframe to dataset.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Dataset as a pandas dataframe.
    feature_cols: list
        List of feature columns to consider.

    """
    
    X = X.unstack(fill_value=np.nan, level=0).stack().swaplevel(0,1)
    samples=len(X.index.unique(level=0))
    timesteps=len(X.index.unique(level=1))
    features=X.shape[1]
    inputs = torch.tensor(X.values.reshape(samples,timesteps,features),dtype=torch.float32)
    target = torch.tensor(y.reshape(len(y),1), dtype=torch.float32)
    return Data(inputs, target)

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)