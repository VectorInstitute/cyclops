"""Utilities for the drift detector module."""

import datetime
import importlib
import inspect
import os
import pickle
from datetime import timedelta
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from alibi_detect.cd import ContextMMDDrift, LearnedKernelDrift
from alibi_detect.utils.pytorch.kernels import DeepKernel, GaussianRBF
from datasets.arrow_dataset import Dataset
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset

from cyclops.models.neural_nets.gru import GRUModel
from cyclops.models.neural_nets.lstm import LSTMModel
from cyclops.models.neural_nets.rnn import RNNModel
from cyclops.models.wrappers import SKModel  # type: ignore
from cyclops.monitor.reductor import Reductor


def print_metrics_binary(
    y_test_labels: Any, y_pred_values: Any, y_pred_labels: Any, verbose: int = 1
) -> Dict[str, Any]:
    """Print metrics for binary classification."""
    conf_matrix = metrics.confusion_matrix(y_test_labels, y_pred_labels)
    if verbose:
        print("confusion matrix:")
        print(conf_matrix)
    conf_matrix = conf_matrix.astype(np.float32)
    tn, fp, fn, tp = conf_matrix.ravel()
    acc = (tn + tp) / np.sum(conf_matrix)
    prec0 = tn / (tn + fn)
    prec1 = tp / (tp + fp)
    rec0 = tn / (tn + fp)
    rec1 = tp / (tp + fn)

    auroc = metrics.roc_auc_score(y_test_labels, y_pred_values)

    (precisions, recalls, _) = metrics.precision_recall_curve(
        y_test_labels, y_pred_values
    )
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print(f"accuracy: {acc}")
        print(f"precision class 0: {prec0}")
        print(f"precision class 1: {prec1}")
        print(f"recall class 0: {rec0}")
        print(f"recall class 1: {rec1}")
        print(f"AUC of ROC: {auroc}")
        print(f"AUC of PRC: {auprc}")
        print(f"min(+P, Se): {minpse}")

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
    }


def load_ckp(
    checkpoint_fpath: str, model: nn.Module
) -> Tuple[nn.Module, Optimizer, int]:
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_fpath)  # type: ignore
    model.load_state_dict(checkpoint["model"])
    optimizer = checkpoint["optimizer"]
    return model, optimizer, checkpoint["n_epochs"]


def get_device() -> torch.device:
    """Get device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_temporal_model(model: str, model_params: Dict[str, Any]) -> nn.Module:
    """Get temporal model.

    Parameters
    ----------
    model: string
        String with model name (e.g. rnn, lstm, gru).

    """
    models = {"rnn": RNNModel, "lstm": LSTMModel, "gru": GRUModel}
    return models[model.lower()](**model_params)


class Data(TorchDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Data class."""

    def __init__(self, inputs: pd.DataFrame, target: pd.DataFrame):
        """Initialize Data class."""
        self.inputs = inputs
        self.target = target

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get item for iterator.

        Parameters
        ----------
        idx: int
            Index of sample to fetch from dataset.

        Returns
        -------
        tuple
            Input and target.

        """
        return self.inputs[idx], self.target[idx]

    def __len__(self) -> int:
        """Return size of dataset, i.e. no. of samples.

        Returns
        -------
        int
            Size of dataset.

        """
        return len(self.target)

    def dim(self) -> Any:
        """Get dataset dimensions (no. of features).

        Returns
        -------
        int
            Number of features.

        """
        return self.inputs.size(dim=1)

    def to_loader(
        self,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = True,
    ) -> DataLoader[Any]:
        """Create dataloader.

        Returns
        -------
            DataLoader with input data

        """
        return DataLoader(
            TensorDataset(self.inputs, self.target),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )


def get_data(X: np.ndarray[float, np.dtype[np.float64]], y: List[int]) -> Data:
    """Convert pandas dataframe to dataset.

    Parameters
    ----------
    X: numpy matrix
        Data containing features in the form of [samples, timesteps, features].
    y: list
        List of labels.

    """
    inputs = torch.tensor(X, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    return Data(inputs, target)


def run_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> SKModel:
    """Choose and run a model on the data and return the best model."""
    if model_name == "mlp":
        model = SKModel("mlp", save_path="./mlp.pkl")
        model.fit(X, y, X_val, y_val)  # pylint: disable=too-many-function-args
    elif model_name == "lr":
        model = SKModel("lr", save_path="./lr.pkl")
        model.fit(X, y, X_val, y_val)  # pylint: disable=too-many-function-args
    elif model_name == "rf":
        model = SKModel("rf", save_path="./rf.pkl")
        model.fit(X, y, X_val, y_val)  # pylint: disable=too-many-function-args
    elif model_name == "xgb":
        model = SKModel("xgb", save_path="./xgb.pkl")
        model.fit(X, y, X_val, y_val)  # pylint: disable=too-many-function-args
    return model


def get_args(obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
            # if key in obj.__code__.co_varnames:
            if key in inspect.getfullargspec(obj).args:
                args[key] = kwargs[key]
    return args


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    """Get object from string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model(model_path: str) -> Any:
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
        model = torch.load(model_path)  # type: ignore
    return model


def save_model(model: Any, output_path: str) -> None:
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
        X_s: np.ndarray[float, np.dtype[np.float64]],
        ds_source: Dataset,
        context_generator: Reductor,
        *,
        backend: str = "tensorflow",
        p_val: float = 0.05,
        preprocess_x_ref: bool = False,
        update_ref: Optional[Dict[str, int]] = None,
        preprocess_fn: Optional[Callable[..., Any]] = None,
        x_kernel: Optional[Callable[..., Any]] = None,
        c_kernel: Optional[Callable[..., Any]] = None,
        n_permutations: int = 1000,
        prop_c_held: float = 0.25,
        n_folds: int = 5,
        batch_size: Optional[int] = 256,
        device: Optional[Union[str, torch.device]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        data_type: Optional[str] = None,
        verbose: bool = False,
    ):
        self.context_generator = context_generator

        c_source = context_generator.transform(ds_source)

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

    def predict(
        self,
        X_t: np.ndarray[float, np.dtype[np.float64]],
        ds_target: Dataset,
        **kwargs: Dict[str, Any],
    ) -> Any:
        """Predict if there is drift in the data."""
        c_target = self.context_generator.transform(ds_target)
        return self.tester.predict(
            X_t, c_target, **get_args(self.tester.predict, kwargs)
        )


class LKWrapper:
    """Wrapper for LKWrapper."""

    def __init__(
        self,
        X_s: np.ndarray[float, np.dtype[np.float64]],
        projection: torch.nn.Module,
        *,
        backend: str = "tensorflow",
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        preprocess_at_init: bool = True,
        update_x_ref: Optional[Dict[str, int]] = None,
        preprocess_fn: Optional[Callable[..., Any]] = None,
        n_permutations: int = 100,
        var_reg: float = 1e-5,
        reg_loss_fn: Callable[..., Any] = (lambda kernel: 0),
        train_size: Optional[float] = 0.75,
        retrain_from_scratch: bool = True,
        optimizer: Optional[Callable[..., Any]] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        preprocess_batch_fn: Optional[Callable[..., Any]] = None,
        epochs: int = 3,
        verbose: int = 0,
        train_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        dataset: Optional[Callable[..., Any]] = None,
        dataloader: Optional[Callable[..., Any]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        data_type: Optional[str] = None,
        kernel_a: nn.Module = GaussianRBF(trainable=True),
        kernel_b: nn.Module = GaussianRBF(trainable=True),
        eps: str = "trainable",
    ):
        self.proj = projection

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

    def predict(
        self, X_t: np.ndarray[float, np.dtype[np.float64]], **kwargs: Dict[str, Any]
    ) -> Any:
        """Predict if there is drift in the data."""
        return self.tester.predict(X_t, **get_args(self.tester.predict, kwargs))


def scale(x: pd.DataFrame) -> pd.DataFrame:
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


def daterange(
    start_date: datetime.date, end_date: datetime.date, stride: int, window: int
) -> Generator[datetime.date, None, None]:
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
    X: pd.DataFrame,
    y: pd.DataFrame,
    admin_data: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
    stride: int = 1,
    window: int = 1,
    ids_to_exclude: Optional[List[str]] = None,
    encounter_id: str = "encounter_id",
    admit_timestamp: str = "admit_timestamp",
) -> Dict[str, Any]:
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


def reshape_2d_to_3d(data: pd.DataFrame, num_timesteps: int) -> pd.DataFrame:
    """Reshape 2D data to 3D data."""
    data = data.unstack()
    num_encounters = data.shape[0]
    data = data.values.reshape((num_encounters, num_timesteps, -1))
    return data


# from https://stackoverflow.com/a/66558182
class Loader:
    """Loaing animation."""

    def __init__(
        self, desc: str = "Loading...", end: str = "Done!", timeout: float = 0.1
    ) -> None:
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

    def start(self) -> "Loader":
        """Start the loader."""
        self._thread.start()
        return self

    def _animate(self) -> None:
        """Animate the loader."""
        for cycle_itr in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {cycle_itr}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self) -> None:
        """Start the thread."""
        self.start()

    def stop(self) -> None:
        """Stop the loader."""
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
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


def nihcxr_preprocess(df: pd.DataFrame, nihcxr_dir: str) -> pd.DataFrame:
    """Preprocess NIHCXR dataframe.

    Add a column with the path to the image and create
    one-hot encoded pathogies from Finding Labels column.

    Parameters
    ----------
        df (pd.DataFrame): NIHCXR dataframe.

    Returns
    -------
        pd.DataFrame: pre-processed NIHCXR dataframe.

    """
    # Add path column
    df["features"] = df["Image Index"].apply(
        lambda x: os.path.join(nihcxr_dir, "images", x)
    )

    # Create one-hot encoded pathologies
    pathologies = df["Finding Labels"].str.get_dummies(sep="|")

    # Add one-hot encoded pathologies to dataframe
    df = pd.concat([df, pathologies], axis=1)

    return df
