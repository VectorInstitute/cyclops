"""Utilities for the drift detector module."""

import importlib
import inspect
import pickle
from datetime import timedelta
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from alibi_detect.cd import ContextMMDDrift, LearnedKernelDrift
from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from torch import nn

from cyclops.monitor.baseline_models.temporal.pytorch.utils import (
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
            # if key in obj.__init__.__code__.co_varnames:
            if key in inspect.signature(obj).parameters:
                args[key] = kwargs[key]
        elif inspect.ismethod(obj) or inspect.isfunction(obj):
            if key in obj.__code__.co_varnames:
                args[key] = kwargs[key]
    return args


def get_obj_from_str(string, reload=False):
    """Get object from string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


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
        backend="pytorch",
        p_val=0.05,
        preprocess_x_ref=True,
        update_ref=None,
        preprocess_fn=None,
        x_kernel=None,
        c_kernel=None,
        n_permutations=100,
        prop_c_held=0.25,
        n_folds=5,
        batch_size=64,
        device=None,
        input_shape=None,
        data_type=None,
        verbose=False,
        context_type="lstm",
        model_path=None,
    ):
        self.context_type = context_type
        self.model_path = model_path

        self.device = device
        if self.device is None:
            self.device = get_device()
        c_source = self.context(X_s)

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
            verbose,
        ]

        self.tester = ContextMMDDrift(X_s, c_source, *args)

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
        backend: str = "tensorflow",
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        preprocess_at_init: bool = True,
        update_x_ref: Optional[Dict[str, int]] = None,
        preprocess_fn: Optional[Callable] = None,
        n_permutations: int = 100,
        var_reg: float = 1e-5,
        reg_loss_fn: Callable = (lambda kernel: 0),
        train_size: Optional[float] = 0.75,
        retrain_from_scratch: bool = True,
        optimizer: Optional[Callable] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        preprocess_batch_fn: Optional[Callable] = None,
        epochs: int = 3,
        verbose: int = 0,
        train_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
        dataset: Optional[Callable] = None,
        dataloader: Optional[Callable] = None,
        input_shape: Optional[tuple] = None,
        data_type: Optional[str] = None,
        kernel_a=GaussianRBF(trainable=True),
        kernel_b=GaussianRBF(trainable=True),
        eps="trainable",
        proj_type="ffnn",
        num_features=None,
        num_classes=None,
    ):

        self.proj = self.choose_proj(X_s, proj_type, num_features, num_classes)

        kernel = DeepKernel(self.proj, kernel_a, kernel_b, eps)

        args = [
            backend,
            p_val,
            x_ref_preprocessed,
            preprocess_at_init,
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
            preprocess_batch_fn,
            epochs,
            verbose,
            train_kwargs,
            device,
            dataset,
            dataloader,
            input_shape,
            data_type,
        ]

        self.tester = LearnedKernelDrift(X_s, kernel, *args)

    def predict(self, X_t, **kwargs):
        """Predict if there is drift in the data."""
        return self.tester.predict(X_t, **get_args(self.tester.predict, kwargs))

    def choose_proj(self, X_s, proj_type, num_features, num_classes):
        """Choose projection for learned kernel drift detection."""
        num_features = num_features or X_s.shape[-1]
        num_classes = num_classes or X_s.shape[-1]
        if proj_type in ["rnn", "gru", "lstm"]:
            proj = recurrent_neural_network(proj_type, num_features, num_classes)
        elif proj_type == "ffnn":
            proj = feed_forward_neural_network(num_features, num_classes)
        elif proj_type == "cnn":
            proj = convolutional_neural_network(num_features, num_classes)
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


def feed_forward_neural_network(
    num_features: int, num_classes: int, ff_dim: int = 256
) -> nn.Module:
    """Create a feed forward neural network model.

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.

    """
    ffnn = nn.Sequential(
        nn.Linear(num_features, ff_dim),
        nn.SiLU(),
        nn.Linear(ff_dim, ff_dim),
        nn.SiLU(),
        nn.Linear(ff_dim, num_classes),
    ).eval()
    return ffnn


def convolutional_neural_network(num_channels, num_classes, cnn_dim=256) -> nn.Module:
    """Create a convolutional neural network model.

    Returns
    -------
    torch.nn.Module
        convolutional neural network for dimensionality reduction.

    """
    cnn = nn.Sequential(
        nn.Conv2d(num_channels, cnn_dim, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(cnn_dim, cnn_dim, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1, -1),
        nn.Linear(cnn_dim, num_classes),
    ).eval()
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


# from https://stackoverflow.com/a/66558182
class Loader:
    """Loaing animation."""

    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """Loader-like context manager.

        Parameters
        ----------
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.

        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        """Start the loader."""
        self._thread.start()
        return self

    def _animate(self):
        """Animate the loader."""
        for cycle_itr in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {cycle_itr}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        """Start the thread."""
        self.start()

    def stop(self):
        """Stop the loader."""
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Stop the thread."""
        # handle exceptions with those variables ^
        self.stop()


if __name__ == "__main__":
    with Loader("Loading with context manager..."):
        for i in range(10):
            sleep(0.25)

    loader = Loader("Loading with object...", "That was fast!", 0.05).start()
    for i in range(10):
        sleep(0.25)
    loader.stop()
