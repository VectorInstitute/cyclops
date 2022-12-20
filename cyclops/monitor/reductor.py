"""Reductor Module."""

import os
import pickle
from typing import Tuple, Union

import numpy as np
import torch
import torchxrayvision as xrv
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cyclops.monitor.utils import get_device, get_temporal_model


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
        - "NoRed"
        - "PCA"
        - "SRP"
        - "kPCA"
        - "Isomap"
        - "GMM"
        - "BBSDs_untrained_FFNN"
        - "BBSDh_untrained_FFNN"
        - "BBSDs_untrained_CNN"
        - "BBSDh_untrained_CNN"
        - "BBSDs_untrained_LSTM"
        - "BBSDh_untrained_LSTM"
        - "BBSDs_trained_LSTM"
        - "BBSDh_trained_LSTM"
        - "BBSDs_txrv_CNN"
        - "BBSDh_txrv_CNN"
        - "TAE_txrv_CNN"
    model_path: str
        The path to the model to use for dimensionality reduction.
        If None, A new model is fit.
    var_ret: float
        The percentage of variance to retain for {"PCA", "SRP", "kPCA", "Isomap"}.
            Note: This uses PCA to determine the number of components
                  but is used for other methods (SRP, kPCA, Isomap).
                  If this behavior is not desired, use n_components directly.
    n_components: int
        The number of components to use for {"PCA", "SRP", "kPCA", "Isomap"}.
        Must be defined for torch datasets.
    gmm_n_clusters: int
        The number of clusters to use for "GMM".
    random_state: int
        The global random seed.

    """

    def __init__(
        self,
        dr_method: str,
        model_path: str = None,
        var_ret: float = 0.8,
        n_components: int = None,
        n_features: int = None,
        n_classes: int = None,
        random_state: int = 42,
    ):

        self.dr_method = dr_method
        self.model_path = model_path
        self.var_ret = var_ret
        self.n_components = n_components
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = get_device()

        self.random_state = random_state
        # set global random seed for all methods
        np.random.seed(self.random_state)

        # dictionary of string methods with corresponding functions
        reductor_methods = {
            "NoRed": None,
            "PCA": PCA,
            "SRP": SparseRandomProjection,
            "kPCA": KernelPCA,
            "Isomap": Isomap,
            "GMM": GaussianMixture,
            "BBSDs_untrained_FFNN": self.feed_forward_neural_network,
            "BBSDh_untrained_FFNN": self.feed_forward_neural_network,
            "BBSDs_untrained_CNN": self.convolutional_neural_network,
            "BBSDh_untrained_CNN": self.convolutional_neural_network,
            "BBSDs_untrained_LSTM": self.recurrent_neural_network,
            "BBSDh_untrained_LSTM": self.recurrent_neural_network,
            "BBSDs_trained_LSTM": self.recurrent_neural_network,
            "BBSDh_trained_LSTM": self.recurrent_neural_network,
            "BBSDs_txrv_CNN": xrv.models.DenseNet,
            "BBSDh_txrv_CNN": xrv.models.DenseNet,
            "TAE_txrv_CNN": xrv.autoencoders.ResNetAE,
        }

        # check if dr_method is valid
        if self.dr_method not in reductor_methods:
            raise ValueError(
                "Invalid dr_method, dr_method must be one of the following: "
                + str(self.get_available_dr_methods())
            )

        if self.dr_method in ("BBSDs_trained_LSTM", "BBSDh_trained_LSTM"):
            self.model = reductor_methods[self.dr_method]("lstm", self.n_features)
            self.model.load_state_dict(torch.load(self.model_path)["model"])
        elif self.dr_method in ("BBSDs_untrained_LSTM", "BBSDh_untrained_LSTM"):
            self.model = reductor_methods[self.dr_method]("lstm", self.n_features)
        elif self.dr_method in ("BBSDs_untrained_FFNN", "BBSDh_untrained_FFNN"):
            self.model = reductor_methods[self.dr_method](
                self.n_features, self.n_classes
            )
        elif self.dr_method in ("BBSDs_untrained_CNN", "BBSDh_untrained_CNN"):
            self.model = reductor_methods[self.dr_method](
                self.n_features, self.n_classes
            )
        elif self.dr_method in ("BBSDs_txrv_CNN", "BBSDh_txrv_CNN"):
            self.model = reductor_methods[self.dr_method](
                weights="densenet121-res224-all"
            )
        elif self.dr_method == "TAE_txrv_CNN":
            self.model = reductor_methods[self.dr_method](weights="101-elastic")
        else:
            self.model = reductor_methods[self.dr_method]

    def load_model(self):
        """Load pre-trained model from path.

        For scikit-learn models, a pickle is loaded from disk. For the torch models, the
        "state_dict" is loaded from disk.

        """
        if self.dr_method in ("PCA", "SRP", "kPCA", "Isomap", "GMM"):
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
        elif self.dr_method in ("BBSDs_trained_LSTM", "BBSDh_trained_LSTM"):
            self.model.load_state_dict(torch.load(self.model_path)["model"])
        print(f"Model loadded from {self.model_path}")

    def save_model(self, output_path: str):
        """Save the model to disk.

        Parameters
        ----------
        output_path: str
            path to save the model to

        """
        if self.dr_method in ("PCA", "SRP", "kPCA", "Isomap", "GMM"):
            with open(output_path, "wb") as file:
                pickle.dump(self.model, file)
        elif self.dr_method in ("BBSDs_trained_LSTM", "BBSDh_trained_LSTM"):
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
            "NoRed",
            "PCA",
            "SRP",
            "kPCA",
            "Isomap",
            "GMM",
            "BBSDs_untrained_FFNN",
            "BBSDh_untrained_FFNN",
            "BBSDs_untrained_CNN",
            "BBSDh_untrained_CNN",
            "BBSDs_untrained_LSTM",
            "BBSDh_untrained_LSTM",
            "BBSDs_trained_LSTM",
            "BBSDh_trained_LSTM",
            "BBSDs_txrv_CNN",
            "TAE_txrv_CNN",
        ]

    def get_dr_amount(self, X: np.ndarray) -> int:
        """Return the number of components to be used to retain the variation specified.

        Returns
        -------
        int
            number of components.

        """
        pca = PCA(n_components=self.var_ret, svd_solver="full")
        pca.fit(X)
        return pca.n_components_

    def fit(self, data: Union[np.ndarray, torch.utils.data.Dataset]):
        """Fit the reductor to the data.

        For pre-trained or untrained models,
        this function loads the weights or initializes the model, respectively.

        Parameters
        ----------
        data: np.ndarray or torch.utils.data.Dataset
            Data to fit the reductor of shape (n_samples, n_features).

        """
        # check if data is a numpy matrix or a torch dataset
        if isinstance(data, np.ndarray):

            if self.dr_method in ("PCA", "SRP", "kPCA", "Isomap", "GMM"):
                if self.n_components is None:
                    self.n_components = self.get_dr_amount(data)

                self.model = self.model(n_components=self.n_components)
                self.model.fit(data)

        elif isinstance(data, torch.utils.data.Dataset):
            pass

        else:
            raise ValueError(
                "data must be a numpy matrix (n_samples, n_features) or a torch Dataset"
            )

    def transform(
        self,
        data: Union[np.ndarray, torch.utils.data.Dataset],
        batch_size: int = 32,
        num_workers: int = None,
        progress: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform the data using the chosen dimensionality reduction method.

        Parameters
        ----------
        data: np.ndarray (n_samples, n_features) or torch Dataset
            data to transform.
        batch_size: int
            batch size for LSTM inference/pytorch dataloader. Default: 32
        num_workers: int
            number of workers for pytorch dataloader. If None, uses max number of cpus.

        Returns
        -------
        X_transformed: numpy.matrix
            transformed data
        optionally: y: numpy.array
            labels of the data

        """
        y = None

        if num_workers is None:
            num_workers = os.cpu_count()

        if self.dr_method in ("PCA", "SRP", "kPCA", "Isomap"):
            if isinstance(data, np.ndarray):
                X_transformed = self.model.transform(data)
            elif isinstance(data, torch.utils.data.Dataset):
                dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed, y = self.batch_inference(
                    self.model, dataloader, progress
                )

        elif self.dr_method == "NoRed":
            if isinstance(data, np.ndarray):
                X_transformed = data
            elif isinstance(data, torch.utils.data.Dataset):
                raise NotImplementedError("NoRed not implemented for torch datasets")
        elif "BBSDs" in self.dr_method:
            if "txrv_CNN" in self.dr_method:
                dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed, y = self.xrv_clf_inference(
                    self.model, dataloader, progress
                )
            else:
                X_transformed = self.minibatch_inference(data, self.model, progress)
        elif "BBSDh" in self.dr_method:
            if "txrv_CNN" in self.dr_method:
                dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed, y = self.xrv_clf_inference(
                    self.model, dataloader, progress
                )
                X_transformed = np.where(X_transformed > 0.5, 1, 0)
            else:
                X_transformed = self.minibatch_inference(data, self.model, progress)
                X_transformed = np.where(X_transformed > 0.5, 1, 0)
        elif self.dr_method == "GMM":
            X_transformed = self.model.predict_proba(data)
        elif self.dr_method == "TAE_txrv_CNN":
            dataloader = DataLoader(
                data, batch_size=batch_size, num_workers=num_workers
            )
            X_transformed, y = self.xrv_ae_inference(self.model, dataloader, progress)
        return X_transformed, y

    def feed_forward_neural_network(
        self, num_features: int, num_classes: int, ff_dim: int = 256
    ) -> nn.Module:
        """Create a feed forward neural network model.

        Returns
        -------
        model: torch.nn.Module
            feed forward neural network model.

        """
        ffnn = (
            nn.Sequential(
                nn.Linear(num_features, ff_dim),
                nn.SiLU(),
                nn.Linear(ff_dim, ff_dim),
                nn.SiLU(),
                nn.Linear(ff_dim, num_classes),
            )
            .to(self.device)
            .eval()
        )
        return ffnn

    def convolutional_neural_network(
        self, num_channels, num_classes, cnn_dim=256
    ) -> nn.Module:
        """Create a convolutional neural network model.

        Returns
        -------
        torch.nn.Module
            convolutional neural network for dimensionality reduction.

        """
        cnn = (
            nn.Sequential(
                nn.Conv2d(num_channels, cnn_dim, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(cnn_dim, cnn_dim, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1, -1),
                nn.Linear(cnn_dim, num_classes),
            )
            .to(self.device)
            .eval()
        )
        return cnn

    def recurrent_neural_network(
        self,
        model_name: str,
        input_dim: int,
        hidden_dim: int = 64,
        layer_dim: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
        last_timestep_only: bool = False,
    ):
        """Create a recurrent neural network model.

        Parameters
        ----------
        model_name: str
            name of the model.
        input_dim: int
            number of input dimensions.
        hidden_dim: int
            number of hidden dimensions.
        layer_dim: int
            number of layers.
        dropout: float
            dropout rate.
        output_dim: int
            number of output dimensions.
        last_timestep_only: bool
            if True, only the last timestep is used as input.

        Returns
        -------
        model: torch.nn.Module
            the rnn model.

        """
        model_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layer_dim": layer_dim,
            "output_dim": output_dim,
            "dropout_prob": dropout,
            "last_timestep_only": last_timestep_only,
        }
        model = get_temporal_model(model_name, model_params).to(self.device).eval()
        return model

    def minibatch_inference(
        self, data, model: nn.Module, batch_size: int = 32
    ) -> np.ndarray:
        """Perform batch inference.

        Performsbatch inferenceon in-memory data by
        breaking into series of mini-batches.

        Parameters
        ----------
        model: torch.nn.Module
            the model to use for inference.

        Returns
        -------
        X_transformed: np.ndarray
            the transformed data.

        """
        if isinstance(data, np.ndarray):
            X = torch.from_numpy(data.astype("float32"))
        num_samples = X.shape[0]
        n_batches = int(np.ceil(num_samples / batch_size))

        X_transformed_all = []
        model.to(self.device)
        with torch.no_grad():
            for i in range(n_batches):
                batch_idx = (
                    i * batch_size,
                    min((i + 1) * batch_size, num_samples),
                )
                X_batch = X[batch_idx[0] : batch_idx[1]]
                X_batch = X_batch.to(self.device)
                X_transformed = model(X_batch)
                if self.device.type == "cuda":
                    X_transformed = X_transformed.cpu()
                X_transformed_all.append(X_transformed.detach().numpy())
        X_transformed = np.concatenate(X_transformed_all, axis=0)
        return X_transformed

    def batch_inference(
        self, model: nn.Module, dataloader: DataLoader, progress=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batched inference on the dataset.

        Parameters
        ----------
        model: torch.nn.Module
            the model to use for inference.

        Returns
        -------
        X_transformed: np.ndarray
            the transformed dataset.
        labels: np.ndarray
            the labels of the transformed dataset.

        """
        imgs_transformed = []
        all_labels = []
        for batch in tqdm(dataloader) if progress else dataloader:
            imgs = batch["img"]
            labels = batch["lab"]
            all_labels.append(labels)
            for img in imgs:
                img_transformed = model.fit_transform(img[0]).flatten()
                imgs_transformed.append(np.expand_dims(img_transformed, axis=0))
        X_transformed = np.concatenate(imgs_transformed)
        labels = np.concatenate(all_labels)
        return X_transformed, labels

    def xrv_clf_inference(
        self, model: nn.Module, dataloader: DataLoader, progress=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batched inference with the TXRV Classifier on the dataset.

        Parameters
        ----------
        model: torch.nn.Module
            the model to use for inference.

        Returns
        -------
        X_transformed: np.ndarray
            the transformed dataset.
        labels: np.ndarray
            the labels of the transformed dataset.

        """
        all_preds = []
        all_labels = []
        model = model.to(self.device).eval()
        for batch in tqdm(dataloader) if progress else dataloader:
            imgs = batch["img"]
            labels = batch["lab"]
            imgs = imgs.to(self.device)
            with torch.no_grad():
                preds = model(imgs)
            preds = preds.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
        X_transformed = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return X_transformed, labels

    def xrv_ae_inference(
        self, model: nn.Module, dataloader: DataLoader, progress=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batched inference with the TXRV Autoencoder on the dataset.

        Parameters
        ----------
        model: torch.nn.Module
            the model to use for inference.

        Returns
        -------
        X_transformed: np.ndarray
            the transformed dataset.
        labels: np.ndarray
            the labels of the transformed dataset.

        """
        all_preds = []
        all_labels = []
        model = model.to(self.device).eval()
        for batch in tqdm(dataloader) if progress else dataloader:
            imgs = batch["img"]
            labels = batch["lab"]
            imgs = imgs.to(self.device)
            with torch.no_grad():
                preds = model.encode(imgs).mean(dim=(2, 3))
            preds = preds.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
        X_transformed = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return X_transformed, labels
