"""Utilities for the drift detector module."""

import datetime
import importlib
import inspect
import pickle
from datetime import timedelta
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from cyclops.models.neural_nets.gru import GRUModel
from cyclops.models.neural_nets.lstm import LSTMModel
from cyclops.models.neural_nets.rnn import RNNModel
from cyclops.models.wrappers import SKModel
from cyclops.utils.optional import import_optional_module


if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data import Dataset as TorchDataset
else:
    torch = import_optional_module("torch", error="warn")
    nn = import_optional_module("torch.nn", error="warn")
    Optimizer = import_optional_module(
        "torch.optim",
        attribute="Optimizer",
        error="warn",
    )
    DataLoader = import_optional_module(
        "torch.utils.data",
        attribute="DataLoader",
        error="warn",
    )
    TensorDataset = import_optional_module(
        "torch.utils.data",
        attribute="TensorDataset",
        error="warn",
    )
    TorchDataset = import_optional_module(
        "torch.utils.data",
        attribute="Dataset",
        error="warn",
    )


def print_metrics_binary(
    y_test_labels: Any,
    y_pred_values: Any,
    y_pred_labels: Any,
    verbose: int = 1,
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
        y_test_labels,
        y_pred_values,
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
    checkpoint_fpath: str,
    model: nn.Module,
) -> Tuple[nn.Module, Optimizer, int]:
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_fpath)  # type: ignore
    model.load_state_dict(checkpoint["model"])
    optimizer = checkpoint["optimizer"]
    return model, optimizer, checkpoint["n_epochs"]


def get_device() -> torch.device:
    """Get device."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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

    def __init__(self, inputs: pd.DataFrame, target: pd.DataFrame) -> None:
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
        model.fit(X, y, X_val, y_val)
    elif model_name == "lr":
        model = SKModel("lr", save_path="./lr.pkl")
        model.fit(X, y, X_val, y_val)
    elif model_name == "rf":
        model = SKModel("rf", save_path="./rf.pkl")
        model.fit(X, y, X_val, y_val)
    elif model_name == "xgb":
        model = SKModel("xgb", save_path="./xgb.pkl")
        model.fit(X, y, X_val, y_val)
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
        if (inspect.isclass(obj) and key in inspect.signature(obj).parameters) or (
            (inspect.ismethod(obj) or inspect.isfunction(obj))
            and key in inspect.getfullargspec(obj).args
        ):
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
    start_date: datetime.date,
    end_date: datetime.date,
    stride: int,
    window: int,
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
        by=admit_timestamp,
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
                (single_date + timedelta(days=window)).strftime("%Y-%m-%d"),
            )
    return {"timestamps": timestamps, "X": X_target_stream, "y": y_target_stream}


def reshape_2d_to_3d(data: pd.DataFrame, num_timesteps: int) -> pd.DataFrame:
    """Reshape 2D data to 3D data."""
    data = data.unstack()
    num_encounters = data.shape[0]
    return data.values.reshape((num_encounters, num_timesteps, -1))


# from https://stackoverflow.com/a/66558182
class Loader:
    """Loaing animation."""

    def __init__(
        self,
        desc: str = "Loading...",
        end: str = "Done!",
        timeout: float = 0.1,
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
        for _i in range(10):
            sleep(0.25)

    loader = Loader("Loading with object...", "That was fast!", 0.05).start()
    for _i in range(10):
        sleep(0.25)
    loader.stop()


class DCELoss(torch.nn.Module):
    """Disagreement Cross Entropy Loss."""

    def __init__(self, weight=None, use_random_vectors=False, alpha=None):
        super(DCELoss, self).__init__()
        self.weight = weight
        self.use_random_vectors = use_random_vectors
        self.alpha = alpha

    def forward(self, logits, labels, mask):
        """Forward pass of the loss function."""
        return dce_loss(
            logits,
            labels,
            mask,
            alpha=self.alpha,
            use_random_vectors=self.use_random_vectors,
            weight=self.weight,
        )


def dce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha: Optional[float] = None,
    use_random_vectors=False,
    weight=None,
) -> torch.Tensor:
    """
    Disagreement Cross Entropy Loss functional.

    :param logits: (batch_size, num_classes) tensor of logits
    :param labels: (batch_size,) tensor of labels
    :param mask: (batch_size,) mask
    :param alpha: (float) weight of q samples
    :param use_random_vectors: (bool) whether to use
           random vectors for negative labels, default=False
    :param weight:  (torch.Tensor) weight for each sample_data,
                    default=None do not apply weighting
    :return: (tensor, float) the disagreement cross entropy loss
    """
    if mask.all():
        # if all labels are positive, then use the standard cross entropy loss
        # infer multi-label classification from the dtype of labels
        if labels.dtype == torch.float32:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss
    if alpha is None:
        alpha = 1 / (1 + (~mask).float().sum())

    num_classes = logits.shape[1]

    q_logits, q_labels = logits[~mask], labels[~mask]
    if use_random_vectors:
        # noinspection PyTypeChecker,PyUnresolvedReferences
        p = -torch.log(
            torch.rand(device=q_labels.device, size=(len(q_labels), num_classes)),
        )
        p *= 1.0 - torch.nn.functional.one_hot(q_labels, num_classes=num_classes)
        p /= torch.sum(p)
        ce_n = -(p * q_logits).sum(1) + torch.logsumexp(q_logits, dim=1)

    else:
        if labels.dtype == torch.long:
            zero_hot = 1.0 - torch.nn.functional.one_hot(
                q_labels,
                num_classes=num_classes,
            )
        else:
            zero_hot = 1.0 - q_labels
        ce_n = -(q_logits * zero_hot).sum(dim=1) / (num_classes - 1) + torch.logsumexp(
            q_logits,
            dim=1,
        )

    if torch.isinf(ce_n).any() or torch.isnan(ce_n).any():
        raise RuntimeError("NaN or Infinite loss encountered for ce-q")

    if (~mask).all():
        return (ce_n * alpha).mean()

    p_logits, p_labels = logits[mask], labels[mask]
    if labels.dtype == torch.float32:
        ce_p = torch.nn.functional.binary_cross_entropy_with_logits(
            p_logits,
            p_labels,
            reduction="none",
            weight=weight,
        )
    else:
        ce_p = torch.nn.functional.cross_entropy(
            p_logits,
            p_labels,
            reduction="none",
            weight=weight,
        )
    return torch.cat([ce_n * alpha, ce_p]).mean()


class DetectronModule(nn.Module):
    """Detectron wrapper module."""

    def __init__(self, model: nn.Module, feature_column: str, alpha=None):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.feature_column = feature_column
        self.criterion = DCELoss(alpha=self.alpha)

    def forward(self, **kwargs):
        """Forward pass of the model."""
        labels = kwargs.pop("labels", None)
        mask = kwargs.pop("mask", None)
        x = kwargs.pop(self.feature_column)
        logits = self.model(x)
        return logits if labels is None else self.criterion(logits, labels, mask)


class DummyCriterion(nn.Module):
    """Dummy criterion."""

    def __init__(self):
        super().__init__()

    def forward(self, loss, labels):
        """Forward pass of the criterion."""
        return loss
