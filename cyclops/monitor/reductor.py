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
from multiprocess import set_start_method

from cyclops.monitor.utils import model_inference


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
            "bbsd-soft+txrv-tae"
    """

    def __init__(
        self,
        dr_method: str,
        device: str = None,
        **kwargs
    ):
        self.dr_method = dr_method
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

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
            "bbse-soft+txrv-tae": BBSE_TAE,
        }

        # check if dr_method is valid
        if self.dr_method not in reductor_methods:
            raise ValueError(
                "Invalid dr_method, dr_method must be one of the following: "
                + str(self.get_available_dr_methods())
            )

        # initialize model
        self.model = reductor_methods[self.dr_method](**kwargs)
        if self.dr_method in ("bbse-soft", "bbse-hard", "txrv-ae", "bbse-soft+txrv-tae"):
            self.model = self.model.to(self.device)

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


    def fit(self, dataset: Dataset):
        """Fit the reductor to the data.

        For pre-trained or untrained models,
        this function loads the weights or initializes the model, respectively.

        Parameters
        ----------
        data: np.ndarray or huggingface Dataset
            Data to fit the reductor of shape (n_samples, n_features).

        """
        # check if data is a numpy matrix or a huggingface dataset
        if self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            self.model.fit(dataset)

    def transform(
        self,
        dataset: Union[Dataset, np.ndarray],
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> np.ndarray:
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
        if num_workers == -1:
            num_workers = os.cpu_count()
        if num_workers > 1:
            try:
                set_start_method("spawn")
            except RuntimeError:
                pass

        if self.dr_method == "NoRed":
            dataset.map(self.nored_inference, batched=True,
                        batch_size=batch_size, num_proc=num_workers)
            features = np.array(dataset["outputs"])
            dataset.remove_columns("outputs")

        elif self.dr_method in ("pca", "srp", "kpca", "isomap", "gmm"):
            if isinstance(dataset, np.ndarray):
                features = self.model.transform(dataset)
            else:
                raise NotImplementedError(
                    "pca, srp, kpca, isomap, gmm not implemented for huggingface datasets, use numpy arrays instead."
                )

        else:
            dataset = dataset.map(self.bbse_inference, batched=True, 
                        batch_size=batch_size, num_proc=num_workers)
            features = np.array(dataset["outputs"])
            dataset.remove_columns("outputs")
        return features
    
    def bbse_inference(self, examples):
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
        images = torch.concat(examples["features"])        
        examples["outputs"] = self.model(images)
        return examples

    def nored_inference(self, examples):
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
        images = torch.stack(examples["features"]).squeeze(1)
        examples["outputs"] = images
        return examples
        

class BlackBoxShiftEstimatorSoft(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x

class BlackBoxShiftEstimatorHard(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model()

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
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tae = TXRVAutoencoder()

    def forward(self, x):
        x_bbse = self.model(x)
        x_tae = self.tae(x)
        x_tae = (x_tae - x_tae.min()) / (x_tae.max() - x_tae.min())
        x = torch.cat((x_bbse, x_tae), dim=1)
        return x
