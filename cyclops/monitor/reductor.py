"""Reductor Module."""

from functools import partial
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torchxrayvision as xrv
from datasets import Dataset, DatasetDict
from monai.transforms import Compose
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torch.utils.data import Dataset as TorchDataset

from cyclops.data.utils import apply_transforms
from cyclops.models.catalog import SKModel, wrap_model


class Reductor:
    """The Reductor class is used to reduce the dimensionality of the data.

    The reductor is initialized with a dimensionality reduction method.
    The reductor can then be fit to the data and used to transform the data.

    Examples
    --------
    >>> # (Data is loaded from memory)
    >>> from drift_detection.reductor import Reductor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reductor = Reductor("pca")
    >>> reductor.fit(X)
    >>> X_transformed = reductor.transform(X)

    Parameters
    ----------
    dr_method: str
        The dimensionality reduction method to use.
        Available methods are:
            "nored"
            "pca"
            "srp"
            "kpca"
            "isomap"
            "gmm"
            "bbsd-soft"
            "bbsd-hard"
            "bbsd-soft+txrv-tae"
    """

    def __init__(
        self,
        dr_method: str,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = None,
        transforms: Optional[Union[Callable, Compose]] = None,
        feature_columns: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        self.dr_method = dr_method.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        if isinstance(transforms, Compose):
            self.transforms = partial(apply_transforms, transforms=transforms)
        else:
            self.transforms = transforms
        self.feature_columns = feature_columns

        # dictionary of string methods with corresponding functions
        reductor_methods = {
            "nored": NoReduction,
            "pca": PCA,
            "srp": SparseRandomProjection,
            "kpca": KernelPCA,
            "isomap": Isomap,
            "gmm": GaussianMixture,
            "bbse-soft": BlackBoxShiftEstimatorSoft,
            "bbse-hard": BlackBoxShiftEstimatorHard,
            "txrv-ae": TXRVAutoencoder,
            "bbse-soft+txrv-ae": BBSETAE,
        }

        # check if dr_method is valid
        if self.dr_method not in reductor_methods:
            raise ValueError(
                "Invalid dr_method, dr_method must be one of the following: "
                + str(self.get_available_dr_methods()),
            )

        # initialize model
        self.model = reductor_methods[self.dr_method](**kwargs)
        if isinstance(self.model, nn.Module):
            self.model = wrap_model(
                self.model,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
            )
            self.model.initialize()
        else:
            self.model = wrap_model(self.model)

    def load_model(self, output_path: str = None) -> None:
        """Load pre-trained model from path.

        For scikit-learn models, a pickle is loaded from disk. For the torch models, the
        "state_dict" is loaded from disk.

        """
        if hasattr(self, "model_path"):
            self.model.load_model(self.model_path)
        elif output_path is not None:
            self.model.load_model(output_path)  # type: ignore
        else:
            raise ValueError("No model path provided.")

    def save_model(self, output_path: str) -> None:
        """Save the model to disk.

        Parameters
        ----------
        output_path: str
            path to save the model to

        """
        self.model.save_model(output_path)
        print(f"{self.dr_method} saved to {output_path}")

    def get_available_dr_methods(self) -> List[str]:
        """Return a list of available dimensionality reduction methods.

        Returns
        -------
        list
            list of available dimensionality reduction methods

        """
        return [
            "nored",
            "pca",
            "srp",
            "kpca",
            "isomap",
            "gmm",
            "bbse-soft",
            "bbse-hard",
            "txrv-ae",
            "bbse-soft+txrv-ae",
        ]

    def fit(self, X: Union[Dataset, DatasetDict, np.ndarray, TorchDataset]) -> None:
        """Fit the reductor to the data.

        For scikit-learn models, the model is fit to the data
        to be used for transforming the data.

        All other methods are pre-trained and do not need to be fit.

        Parameters
        ----------
        dataset: huggingface Dataset
            dataset to fit the reductor to.

        """
        if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            self.model.fit(
                X,
                transforms=self.transforms,
                feature_columns=self.feature_columns,
                dim_reduction=True,
            )

    def transform(
        self,
        X: Union[Dataset, DatasetDict, np.ndarray, TorchDataset],
    ) -> np.ndarray[float, np.dtype[np.float32]]:
        """Transform the data using the chosen dimensionality reduction method.

        Parameters
        ----------
        dataset: huggingface Dataset
            data to transform.
        batch_size: int
            batch size for huggingface map operation. Default: 32
        num_workers: int
            number of workers for huggingface map operation.
            If -1, uses max number of cpus. Default: 1

        Returns
        -------
        features: np.ndarray
            transformed data

        """
        if isinstance(self.model, SKModel) and self.dr_method != "gmm":
            X_reduced = self.model.predict(
                X,
                transforms=self.transforms,
                feature_columns=self.feature_columns,
                only_predictions=True,
                dim_reduction=True,
            )
        elif self.dr_method == "gmm":
            X_reduced = self.model.predict_proba(
                X,
                transforms=self.transforms,
                feature_columns=self.feature_columns,
                only_predictions=True,
            )
        else:
            X_reduced = self.model.predict(
                X,
                transforms=self.transforms,
                feature_columns=self.feature_columns,
                only_predictions=True,
            )
        return np.array(X_reduced).astype("float32")


class NoReduction(BaseEstimator):
    """No reduction dummy function."""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: np.ndarray) -> None:
        """Fit the model."""
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Predict the model."""
        return X


class BlackBoxShiftEstimatorSoft(nn.Module):
    """Wrapper for Black Box Shift Estimator Soft model."""

    def __init__(self, model: nn.Module, softmax: bool = False) -> None:
        super().__init__()
        self.model = model
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.model(x)
        if self.softmax:
            x = torch.softmax(x, dim=-1)
        return x


class BlackBoxShiftEstimatorHard(nn.Module):
    """Wrapper for Black Box Shift Estimator Hard model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.model(x)
        return x.argmax(dim=-1)


class TXRVAutoencoder(nn.Module):
    """Wrapper for TXRV Autoencoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.model = xrv.autoencoders.ResNetAE("101-elastic")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model.encode(x).mean(dim=(2, 3))


class BBSETAE(nn.Module):
    """Wrapper for Black Box Shift Estimator Soft + TXRV Autoencoder model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.tae = TXRVAutoencoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x_bbse = self.model(x)
        x_tae = self.tae(x)
        x_tae = (x_tae - x_tae.min()) / (x_tae.max() - x_tae.min())
        return torch.cat((x_bbse, x_tae), dim=1)
