"""Utilities for the drift detector module."""

import inspect
import pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from alibi_detect.cd import ContextMMDDrift, LearnedKernelDrift
from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from torch import nn

from drift_detection.baseline_models.temporal.pytorch.utils import (
    get_device,
    get_temporal_model,
)


def get_args(obj, kwargs):
    """Get valid arguments from kwargs to pass to object.

    Parameters
    ----------
    obj
        object to get arguments from.
    kwargs
        Dictionary of arguments to pass to object.

    Returns
    -------
    args
        Dictionary of valid arguments to pass to class object.

    """
    args = {}
    for key in kwargs:
        if inspect.isclass(obj):
            if key in obj.__init__.__code__.co_varnames:
                args[key] = kwargs[key]
        elif inspect.ismethod(obj) or inspect.isfunction(obj):
            if key in obj.__code__.co_varnames:
                args[key] = kwargs[key]
    return args


def load_model(model_path: str):
    """Load pre-trained model from path.

    Loads pre-trained model from specified model path.
    For scikit-learn models, a pickle is loaded from disk.
    For the pytorch models, the "state_dict" is loaded from disk.

    Returns
    -------
    model
        loaded pre-trained model

    """
    file_type = model_path.split(".")[-1]
    if file_type in ("pkl", "pickle"):
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    elif file_type == "pt":
        model = torch.load(model_path)
    return model


def save_model(model, output_path: str):
    """Save the model to disk.

    For scikit-learn models, a pickle is saved to disk.
    For the pytorch models, the "state_dict" is saved to disk.

    Parameters
    ----------
    output_path: String
        path to save the model to

    """
    file_type = output_path.split(".")[-1]
    if file_type in ("pkl", "pickle"):
        with open(output_path, "wb") as file:
            pickle.dump(model, file)
    elif file_type == "pt":
        torch.save(model.state_dict(), output_path)

class ContextMMDWrapper:
    """Wrapper for ContextMMDDrift."""
    
    def __init__(
        self, 
        X_s, 
        *, 
        backend= 'pytorch', 
        p_val = 0.05, 
        preprocess_x_ref = True, 
        update_ref = None, 
        preprocess_fn = None, 
        x_kernel = None, 
        c_kernel = None, 
        n_permutations = 100, 
        prop_c_held = 0.25, 
        n_folds = 5, 
        batch_size = 64, 
        device = None, 
        input_shape = None, 
        data_type = None, 
        verbose = False, 
        context_type='lstm', 
        model_path=None
    ):
        self.context_type = context_type
        self.model_path = model_path
        self.device = device
        if self.device is None:
            self.device = get_device()    
        C_s = self.context(X_s)

        args = [
            backend, 
            p_val, 
            preprocess_x_ref, 
            update_ref, 
            preprocess_fn, 
            x_kernel, 
            c_kernel, 
            n_permutations, 
            prop_c_held, 
            n_folds,
            batch_size, 
            device, 
            input_shape, 
            data_type, 
            verbose
        ]
        
        self.tester = ContextMMDDrift(X_s, C_s, *args)

    def predict(self, X_t, **kwargs):
        """Predict if there is drift in the data."""
        c_target = self.context(X_t)
        return self.tester.predict(
            X_t, c_target, **get_args(self.tester.predict, kwargs)
        )

    def context(self, X: np.ndarray):
        """Get context for context mmd drift detection.

        Parameters
        ----------
        X
            Data to build context for context mmd drift detection.

        """
        if self.context_type == "sklearn":
            model = load_model(self.model_path)
            pred_proba = model.predict_proba(X)
            output = pred_proba
        elif self.context_type in ["rnn", "gru", "lstm"]:
            model = recurrent_neural_network(self.context_type, X.shape[-1])
            model.load_state_dict(load_model(self.model_path)["model"])
            model.to(self.device).eval()
            with torch.no_grad():
                output = model(torch.from_numpy(X).to(self.device)).cpu().numpy()
                output = softmax(output, -1)
        else:
            raise ValueError("Context not supported")
        return output

class LKWrapper:
    """Wrapper for LKWrapper."""

    def __init__(
        self,
        X_s,
        *,
        backend="pytorch",
        p_val=0.05,
        preprocess_x_ref=True,
        update_x_ref=None,
        preprocess_fn=None,
        n_permutations=100,
        var_reg=0.00001,
        reg_loss_fn=lambda kernel: 0,
        train_size=0.75,
        retrain_from_scratch=True,
        optimizer=None,
        learning_rate=0.001,
        batch_size=32,
        preprocess_batch=None,
        epochs=3,
        verbose=0,
        train_kwargs=None,
        device=None,
        dataset=None,
        dataloader=None,
        data_type=None,
        kernel_a=GaussianRBF(trainable=True),
        kernel_b=GaussianRBF(trainable=True),
        eps="trainable",
        proj_type="ffnn"
    ):

        self.proj = self.choose_proj(X_s, proj_type)

        kernel = DeepKernel(self.proj, kernel_a, kernel_b, eps)

        args = [
            backend,
            p_val,
            preprocess_x_ref,
            update_x_ref,
            preprocess_fn,
            n_permutations,
            var_reg,
            reg_loss_fn,
            train_size,
            retrain_from_scratch,
            optimizer,
            learning_rate,
            batch_size,
            preprocess_batch,
            epochs,
            verbose,
            train_kwargs,
            device,
            dataset,
            dataloader,
            data_type,
        ]

        self.tester = LearnedKernelDrift(X_s, kernel, *args)

    def predict(self, X_t, **kwargs):
        """Predict if there is drift in the data."""
        return self.tester.predict(X_t, **get_args(self.tester.predict, kwargs))

    def choose_proj(self, X_s, proj_type):
        """Choose projection for learned kernel drift detection."""
        if proj_type in ["rnn", "gru", "lstm"]:
            proj = recurrent_neural_network(proj_type, X_s.shape[-1])
        elif proj_type == "ffnn":
            proj = feed_forward_neural_network(X_s.shape[-1])
        elif proj_type == "cnn":
            proj = convolutional_neural_network(X_s.shape[-1])
        else:
            raise ValueError("Invalid projection type.")
        return proj


def recurrent_neural_network(
    model_name: str,
    input_dim: int,
    hidden_dim=64,
    layer_dim=2,
    dropout=0.2,
    output_dim=1,
    last_timestep_only=False,
):
    """Create a recurrent neural network model.

    Parameters
    ----------
    model_name
        type of rnn model, one of: "rnn", "lstm", "gru"
    input_dim
        number of features

    Returns
    -------
    model: torch.nn.Module
        recurrent neural network model.

    """
    model_params = {
        "device": get_device(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layer_dim": layer_dim,
        "output_dim": output_dim,
        "dropout_prob": dropout,
        "last_timestep_only": last_timestep_only,
    }
    model = get_temporal_model(model_name, model_params)
    return model


def feed_forward_neural_network(input_dim: int):
    """Create a feed forward neural network model.

    Parameters
    ----------
    input_dim
        number of features

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.

    """
    ffnn = nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.SiLU(),
        nn.Linear(16, 8),
        nn.SiLU(),
        nn.Linear(8, 1),
    )
    return ffnn


def convolutional_neural_network(input_dim: int):
    """Create a convolutional neural network model.

    Parameters
    ----------
    input_dim
        number of features

    Returns
    -------
    torch.nn.Module
        convolutional neural network.

    """
    cnn = nn.Sequential(
        nn.Conv2d(input_dim, 4, 3, 2, 0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, 3, 2, 0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, 3, 2, 0),
        nn.ReLU(),
        nn.Flatten(),
    )
    return cnn


def scale(x: pd.DataFrame):
    """Scale columns of temporal dataframe.

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.

    """
    numerical_cols = [
        col for col in x if not np.isin(x[col].dropna().unique(), [0, 1]).all()
    ]

    for col in numerical_cols:
        scaler = StandardScaler().fit(x[col].values.reshape(-1, 1))
        x[col] = pd.Series(
            np.squeeze(scaler.transform(x[col].values.reshape(-1, 1))),
            index=x[col].index,
        )

    return x


def daterange(start_date, end_date, stride: int, window: int):
    """Output a range of dates.

    Outputs a range of dates after applying a shift of
    a given stride and window adjustment.

    Returns
    -------
    datetime.date
        range of dates after stride and window adjustment.

    """
    for date in range(int((end_date - start_date).days)):
        if start_date + timedelta(date * stride + window) < end_date:
            yield start_date + timedelta(date * stride)


def get_serving_data(
    X,
    y,
    admin_data,
    start_date,
    end_date,
    stride=1,
    window=1,
    ids_to_exclude=None,
    encounter_id="encounter_id",
    admit_timestamp="admit_timestamp",
):
    """Transform a static set of patient encounters with timestamps into serving data.

    Transforms a static set of patient encounters with timestamps into
    serving data that ranges from a given start date and goes until
    a given end date with a constant window and stride length.

    Returns
    -------
    dictionary
        dictionary containing keys timestamp, X and y

    """
    X_target_stream = []
    y_target_stream = []
    timestamps = []

    admit_df = admin_data[[encounter_id, admit_timestamp]].sort_values(
        by=admit_timestamp
    )
    for single_date in daterange(start_date, end_date, stride, window):
        if single_date.month == 1 and single_date.day == 1:
            print(
                single_date.strftime("%Y-%m-%d"),
                "-",
                (single_date + timedelta(days=window)).strftime("%Y-%m-%d"),
            )
        encounters_inwindow = admit_df.loc[
            (
                (single_date + timedelta(days=window)).strftime("%Y-%m-%d")
                > admit_df[admit_timestamp].dt.strftime("%Y-%m-%d")
            )
            & (
                admit_df[admit_timestamp].dt.strftime("%Y-%m-%d")
                >= single_date.strftime("%Y-%m-%d")
            ),
            encounter_id,
        ].unique()
        if ids_to_exclude is not None:
            encounters_inwindow = [
                x for x in encounters_inwindow if x not in ids_to_exclude
            ]
        encounter_ids = X.index.get_level_values(0).unique()
        X_inwindow = X.loc[X.index.get_level_values(0).isin(encounters_inwindow)]
        y_inwindow = pd.DataFrame(y[np.in1d(encounter_ids, encounters_inwindow)])
        if not X_inwindow.empty:
            X_target_stream.append(X_inwindow)
            y_target_stream.append(y_inwindow)
            timestamps.append(
                (single_date + timedelta(days=window)).strftime("%Y-%m-%d")
            )
    target_data = {"timestamps": timestamps, "X": X_target_stream, "y": y_target_stream}
    return target_data


def reshape_2d_to_3d(data, num_timesteps):
    """Reshape 2D data to 3D data."""
    data = data.unstack()
    num_encounters = data.shape[0]
    data = data.values.reshape((num_encounters, num_timesteps, -1))
    return data
