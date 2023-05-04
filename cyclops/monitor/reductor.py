"""Reductor Module."""

import pickle
from multiprocessing import set_start_method
from typing import Any, Dict, List

import numpy as np
import torch
import torchxrayvision as xrv
from datasets.arrow_dataset import Dataset
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from torch import nn


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

    def __init__(self, dr_method: str, device: str = "cpu", **kwargs: Any):
        self.dr_method = dr_method.lower()
        self.device = device

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
                + str(self.get_available_dr_methods())
            )

        # initialize model
        self.model = reductor_methods[self.dr_method](**kwargs)

    def load_model(self) -> None:
        """Load pre-trained model from path.

        For scikit-learn models, a pickle is loaded from disk. For the torch models, the
        "state_dict" is loaded from disk.

        """
        if hasattr(self, "model_path"):
            if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
                with open(self.model_path, "rb") as file:
                    self.model = pickle.load(file)
            else:
                self.model.load_state_dict(torch.load(self.model_path))  # type: ignore
            print(f"Model loadded from {self.model_path}")
        else:
            raise ValueError("model_path not set.")

    def save_model(self, output_path: str) -> None:
        """Save the model to disk.

        Parameters
        ----------
        output_path: str
            path to save the model to

        """
        if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            with open(output_path, "wb") as file:
                pickle.dump(self.model, file)
        else:
            torch.save(self.model.state_dict(), output_path)
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

    def fit(self, dataset: Dataset) -> None:
        """Fit the reductor to the data.

        For scikit-learn models, the model is fit to the data
        to be used for transforming the data.

        All other methods are pre-trained and do not need to be fit.

        Parameters
        ----------
        dataset: huggingface Dataset
            dataset to fit the reductor to.

        """
        features = dataset["features"]
        if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            self.model.fit(features)

    def transform(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> np.ndarray[float, np.dtype[np.float64]]:
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
        if num_workers > 1:
            try:
                set_start_method("spawn")
            except RuntimeError:
                pass

        if self.dr_method == "nored":
            dataset = dataset.map(
                self.nored_inference,
                batched=True,
                batch_size=batch_size,
                num_proc=num_workers,
            )
            features = np.array(dataset["outputs"])
            dataset.remove_columns("outputs")

        elif self.dr_method in ("pca", "srp", "kpca", "isomap"):
            features = self.model.transform(dataset["features"])
        elif self.dr_method in ("gmm"):
            features = self.model.predict_proba(dataset["features"])
        else:
            self.model = self.model.to(self.device)
            dataset = dataset.map(
                self.bbse_inference,
                batched=True,
                batch_size=batch_size,
                num_proc=num_workers,
            )
            features = np.array(dataset["outputs"])
            dataset.remove_columns("outputs")
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
        return features

    def bbse_inference(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Inference function for Black Box Shift Estimator models.

        Parameters
        ----------
        examples: dict
            dictionary containing the data to use for inference.

        Returns
        -------
        examples: dict
            dictionary containing the data with the outputs.

        """
        if isinstance(examples["features"][0], np.ndarray):
            features = np.concatenate(examples["features"])
        elif isinstance(examples["features"][0], torch.Tensor):
            features = torch.concat(examples["features"])
        elif isinstance(examples["features"][0], list):
            if isinstance(self.model, torch.nn.Module):
                features = torch.tensor(examples["features"])
            else:
                features = np.array(examples["features"])
        examples["outputs"] = self.model(features)
        return examples

    def nored_inference(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Inference function for no dimensionality reduction.

        Parameters
        ----------
        examples: dict
            dictionary containing the data to use for inference.

        Returns
        -------
        examples: dict
            dictionary containing the data with the outputs.

        """
        if isinstance(examples["features"][0], np.ndarray):
            features = np.concatenate(examples["features"])
        elif isinstance(examples["features"][0], torch.Tensor):
            features = torch.concat(examples["features"])
        elif isinstance(examples["features"][0], list):
            features = np.array(examples["features"])
        examples["outputs"] = features
        return examples


class NoReduction:
    """No reduction dummy function."""

    def __init__(self, reduce: Any = None) -> None:
        self.reduce = reduce

    def fit(self, x: Any) -> None:
        """Run dummy fit function."""


class BlackBoxShiftEstimatorSoft(nn.Module):
    """Wrapper for Black Box Shift Estimator Soft model."""

    def __init__(self, model: nn.Module, softmax: bool = False):
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

    def __init__(self, model: nn.Module):
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
        x = self.model.encode(x).mean(dim=(2, 3))
        return x


class BBSETAE(nn.Module):
    """Wrapper for Black Box Shift Estimator Soft + TXRV Autoencoder model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.tae = TXRVAutoencoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x_bbse = self.model(x)
        x_tae = self.tae(x)
        x_tae = (x_tae - x_tae.min()) / (x_tae.max() - x_tae.min())
        x = torch.cat((x_bbse, x_tae), dim=1)
        return x
