"""Reductor Module."""

import os
import pickle
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchxrayvision as xrv
from datasets.arrow_dataset import Dataset
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cyclops.monitor.utils import get_device
from cyclops.monitor.utils import minibatch_inference, batch_inference, xrv_inference


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
    >>> reductor = Reductor("PCA")
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
            "bbsd-soft+tae"
    """

    def __init__(
        self,
        dr_method: str,
        **kwargs
    ):
        self.dr_method = dr_method

        # dictionary of string methods with corresponding functions
        reductor_methods = {
            "nored": None,
            "pca": PCA,
            "srp": SparseRandomProjection,
            "kpca": KernelPCA,
            "isomap": Isomap,
            "gmm": GaussianMixture,
            "bbse-soft": BlackBoxShiftEstimatorSoft,
            "bbse-hard": BlackBoxShiftEstimatorHard,
            "txrv-ae": TXRVAutoencoder,
            "bbse-soft+txrv-ae": BBSE_TAE,
        }

        # check if dr_method is valid
        if self.dr_method not in reductor_methods:
            raise ValueError(
                "Invalid dr_method, dr_method must be one of the following: "
                + str(self.get_available_dr_methods())
            )

        # initialize model
        self.model = reductor_methods[self.dr_method](**kwargs)

    def load_model(self):
        """Load pre-trained model from path.

        For scikit-learn models, a pickle is loaded from disk. For the torch models, the
        "state_dict" is loaded from disk.

        """
        if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
        else:
            self.model.load_state_dict(torch.load(self.model_path)["model"])
        print(f"Model loadded from {self.model_path}")

    def save_model(self, output_path: str):
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

    def get_available_dr_methods(self):
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
            "bbse-soft+txrv-ae"
        ]


    def fit(self, data: Union[np.ndarray, Dataset]):
        """Fit the reductor to the data.

        For pre-trained or untrained models,
        this function loads the weights or initializes the model, respectively.

        Parameters
        ----------
        data: np.ndarray or huggingface Dataset
            Data to fit the reductor of shape (n_samples, n_features).

        """
        # check if data is a numpy matrix or a huggingface dataset
        if isinstance(data, np.ndarray):
            if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
                self.model.fit(data)

        elif isinstance(data, Dataset):
            pass

        else:
            raise ValueError(
                "data must be a numpy matrix (n_samples, n_features) \
                     or a huggingface Dataset"
            )

    def transform(
        self,
        data: Union[np.ndarray, Dataset],
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        progress: bool = True,
        device: str = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform the data using the chosen dimensionality reduction method.

        Parameters
        ----------
        data: np.ndarray (n_samples, n_features) or huggingface Dataset
            data to transform.
        batch_size: int
            batch size for pytorch dataloader. Default: 32
        num_workers: int
            number of workers for pytorch dataloader. If None, uses max number of cpus.

        Returns
        -------
        X_transformed: numpy.matrix
            transformed data
        
        """
        if num_workers is None:
            num_workers = os.cpu_count()
        
        if self.dr_method == "NoRed":
            if isinstance(data, np.ndarray):
                X_transformed = data
            elif isinstance(data, torch.utils.data.Dataset):
                raise NotImplementedError("NoRed not implemented for torch datasets")

        elif self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            if isinstance(data, np.ndarray):
                X_transformed = self.model.transform(data)
            elif isinstance(data, Dataset):
                dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed = batch_inference(
                    self.model, dataloader, progress, device
                )

        elif self.dr_method == "GMM":
            X_transformed = self.model.predict_proba(data)

        else:
            if isinstance(data, np.ndarray):
                X_transformed = minibatch_inference(data, self.model, batch_size, device)
            elif isinstance(data, torch.utils.data.Dataset):
                dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed = xrv_inference(
                    self.model, dataloader, progress, device
                )
        return X_transformed
        

class BlackBoxShiftEstimatorSoft(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model(**kwargs)

    def forward(self, x):
        x = self.model(x)
        return x.softmax(dim=-1)

class BlackBoxShiftEstimatorHard(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model(**kwargs)

    def forward(self, x):
        x = self.model(x)
        return x.argmax(dim=-1)

class TXRVAutoencoder(nn.Module):
    def __init__(self, weights="101-elastic"):
        super().__init__()
        self.model = xrv.autoencoders.ResNetAE(weights)

    def forward(self, x):
        x = self.model.encode(x).mean(dim=(2, 3))
        return x

class BBSE_TAE(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model(**kwargs)
        self.tae = TXRVAutoencoder()

    def forward(self, x):
        x_bbse = self.model(x)
        x_bbse = x_bbse.softmax(dim=-1)
        x_tae = self.tae(x)
        x_tae = (x_tae - x_tae.min()) / (x_tae.max() - x_tae.min())
        x = torch.cat((x_bbse, x_tae), dim=1)
        return x
