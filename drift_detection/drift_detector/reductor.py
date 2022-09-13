import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from typing import Optional, List, Union, Tuple, Dict, Any

sys.path.append("..")

from baseline_models.temporal.pytorch.optimizer import Optimizer
from baseline_models.temporal.pytorch.utils import (
    get_temporal_model,
    get_device,
)


class Reductor:

    """
    The Reductor class is used to reduce the dimensionality of the data.
    The reductor is initialized with a dimensionality reduction method.
    The reductor can then be fit to the data and used to transform the data.
    Example: (Data is loaded from memory)
    --------
    >>> from drift_detection.reductor import Reductor
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> reductor = Reductor("PCA")
    >>> reductor.fit(X)
    >>> X_transformed = reductor.transform(X)
    Arguments
    ---------
    dr_method: String
        The dimensionality reduction method to use.
        Available methods are:
            "PCA"
            "SRP"
            "kPCA"
            "Isomap"
            "GMM"
            "BBSDs_untrained_FFNN"
            "BBSDh_untrained_FFNN"
            "BBSDs_untrained_CNN"
            "BBSDh_untrained_CNN"
            "BBSDs_untrained_LSTM"
            "BBSDh_untrained_LSTM"
            "BBSDs_trained_LSTM"
            "BBSDh_trained_LSTM"
            "BBSDs_txrv_CNN"
            "BBSDh_txrv_CNN"
            "TAE_txrv_CNN"
    model_path: String
        The path to the model to use for dimensionality reduction.
        If None, A new model is fit.
    var_ret: float
        The percentage of variance to retain for {"PCA", "SRP", "kPCA", "Isomap"}.
            Note: This uses PCA to determine the number of components but is used for other methods (SRP, kPCA, Isomap).
                  If this behavior is not desired, use n_components directly.
    n_components: int
        The number of components to use for {"PCA", "SRP", "kPCA", "Isomap"}.  Must be defined for torch datasets.
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
        gmm_n_clusters: int = 2,
        random_state: int = 42,
    ):

        self.dr_method = dr_method
        self.model_path = model_path
        self.var_ret = var_ret
        self.n_components = n_components
        self.n_features = n_features
        self.gmm_n_clusters = gmm_n_clusters
        self.device = get_device()

        self.random_state = random_state
        np.random.seed(self.random_state)  # set global random seed for all methods

        # dictionary of string methods with corresponding functions
        reductor_methods = {
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
        if self.dr_method not in reductor_methods.keys():
            raise ValueError(
                "Invalid dr_method, dr_method must be one of the following: "
                + str(self.get_available_dr_methods())
            )

        if (
            self.dr_method == "BBSDs_trained_LSTM"
            or self.dr_method == "BBSDh_trained_LSTM"
        ):
            self.model = reductor_methods[self.dr_method]("lstm", self.n_features)
            self.model.load_state_dict(torch.load(self.model_path))
        elif (
            self.dr_method == "BBSDs_Untrained_LSTM"
            or self.dr_method == "BBSDh_Untrained_LSTM"
        ):
            self.model = reductor_methods[self.dr_method]("lstm", self.n_features)
        elif (
            self.dr_method == "BBSDs_untrained_FFNN"
            or self.dr_method == "BBSDh_untrained_FFNN"
        ):
            self.model = reductor_methods[self.dr_method](self.n_features)
        elif self.dr_method == "BBSDs_txrv_CNN" or self.dr_method == "BBSDh_txrv_CNN":
            self.model = reductor_methods[self.dr_method](
                weights="densenet121-res224-all"
            )
        elif self.dr_method == "TAE_txrv_CNN":
            self.model = reductor_methods[self.dr_method](weights="101-elastic")
        else:
            self.model = reductor_methods[self.dr_method]

    def load_model(self):
        """Load pre-trained model from path.
        For scikit-learn models, a pickle is loaded from disk.
        For the torch models, the "state_dict" is loaded from disk.
        """
        if (
            self.dr_method == "PCA"
            or self.dr_method == "SRP"
            or self.dr_method == "kPCA"
            or self.dr_method == "Isomap"
            or self.dr_method == "GMM"
        ):
            self.model = pickle.load(open(self.model_path, "rb"))
        elif (
            self.dr_method == "BBSDs_trained_LSTM"
            or self.dr_method == "BBSDh_trained_LSTM"
        ):
            self.model.load_state_dict(torch.load(self.model_path))
        print("Model loaded from {}".format(self.model_path))

    def save_model(self, output_path: str):
        """Saves the model to disk.
        Parameters
        ----------
        output_path: String
            path to save the model to
        """
        if (
            self.dr_method == "PCA"
            or self.dr_method == "SRP"
            or self.dr_method == "kPCA"
            or self.dr_method == "Isomap"
            or self.dr_method == "GMM"
        ):
            pickle.dump(self.model, open(output_path, "wb"))
        elif (
            self.dr_method == "BBSDs_trained_LSTM"
            or self.dr_method == "BBSDh_trained_LSTM"
        ):
            torch.save(self.model.state_dict(), output_path)
        print("{} saved to {}".format(self.dr_method, output_path))

    def get_available_dr_methods(self):
        """Returns a list of available dimensionality reduction methods.
        Returns
        -------
        list
            list of available dimensionality reduction methods
        """
        return [
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
        """
        Returns the number of components to be used to retain the variation specified.
        Returns
        -------
        int
            number of components.
        """
        pca = PCA(n_components=self.var_ret, svd_solver="full")
        pca.fit(X)
        return pca.n_components_

    def fit(self, data: Union[np.ndarray, torch.utils.data.Dataset]):
        """Fits the reductor to the data.
        For pre-trained or untrained models, this function loads the weights or initializes the model, respectively.
        Parameters
        ----------
        data:
            data to fit the reductor to.
            Shape:
                data: np.ndarray (n_samples, n_features)
                      or torch Dataset
        """
        # check if data is a numpy matrix or a torch dataset
        if isinstance(data, np.ndarray):

            if self.n_components is None:
                self.n_components = self.get_dr_amount(data)

            if (
                self.dr_method == "PCA"
                or self.dr_method == "SRP"
                or self.dr_method == "kPCA"
                or self.dr_method == "Isomap"
                or self.dr_method == "GMM"
            ):
                self.model = self.model(n_components=self.n_components)
                self.model.fit(data)

        elif isinstance(data, torch.utils.data.Dataset):
            raise ValueError(
                "fit() does not perform any operations on torch Datasets. Use just transform() instead."
            )

        else:
            raise ValueError(
                "data must be a numpy matrix (n_samples, n_features) or a torch Dataset"
            )

    def transform(
        self,
        data: Union[np.ndarray, torch.utils.data.Dataset],
        batch_size: int = 32,
        num_workers: int = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transforms the data using the chosen dimensionality reduction method
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
        """
        y = None

        self.batch_size = batch_size
        if num_workers is None:
            num_workers = os.cpu_count()

        if (
            self.dr_method == "PCA"
            or self.dr_method == "SRP"
            or self.dr_method == "kPCA"
            or self.dr_method == "Isomap"
        ):
            if isinstance(data, np.ndarray):
                X_transformed = self.model.transform(data)
            elif isinstance(data, torch.utils.data.Dataset):
                self.dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed, y = self.batch_inference(self.model)

        elif self.dr_method == "NoRed":
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, torch.utils.data.Dataset):
                return NotImplementedError("NoRed not implemented for torch datasets")
        elif "BBSDs" in self.dr_method:
            if "txrv_CNN" in self.dr_method:
                self.dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed, y = self.xrv_clf_inference(self.model)
            else:
                X_transformed = self.minibatch_inference(self.model)
        elif "BBSDh" in self.dr_method:
            if "txrv_CNN" in self.dr_method:
                self.dataloader = DataLoader(
                    data, batch_size=batch_size, num_workers=num_workers
                )
                X_transformed, y = self.xrv_clf_inference(self.model)
                X_transformed = np.where(X_transformed > 0.5, 1, 0)
            else:
                X_transformed = self.minibatch_inference(self.model)
                X_transformed = np.where(X_transformed > 0.5, 1, 0)
        elif self.dr_method == "GMM":
            X_transformed = self.model.predict_proba(data)
        elif self.dr_method == "TAE_txrv_CNN":
            self.dataloader = DataLoader(
                data, batch_size=batch_size, num_workers=num_workers
            )
            X_transformed, y = self.xrv_ae_inference(self.model)
        if y is not None:
            return X_transformed, y
        else:
            return X_transformed

    def feed_forward_neural_network(self, n_features: int):
        """
        Creates a feed forward neural network model.
        Returns
        -------
        model: torch.nn.Module
            feed forward neural network model.
        """
        ffnn = (
            nn.Sequential(
                nn.Linear(n_features, 16),
                nn.SiLU(),
                nn.Linear(16, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
            )
            .to(self.device)
            .eval()
        )
        return ffnn

    def convolutional_neural_network(self):
        """
        Creates a convolutional neural network model.
        Returns
        -------
        torch.nn.Module
            convolutional neural network for dimensionality reduction.
        """
        cnn = (
            nn.Sequential(
                nn.Conv2d(3, 8, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(8, 16, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
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
        dropout: int = 0.2,
        output_dim: int = 1,
        last_timestep_only: bool = False,
    ):
        """
        Creates a recurrent neural network model.
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
            "device": self.device,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layer_dim": layer_dim,
            "output_dim": output_dim,
            "dropout_prob": dropout,
            "last_timestep_only": last_timestep_only,
        }
        model = get_temporal_model(model_name, model_params).to(self.device).eval()
        return model

    def gaussian_mixture_model(self):
        """
        Creates a gaussian mixture model.
        """
        gmm = GaussianMixture(n_components=self.gmm_n_clusters, covariance_type="full")
        return gmm

    def minibatch_inference(self, model: nn.Module) -> np.ndarray:
        """
        Performs batch inference on in-memory data by breaking into series of mini-batches.
        Parameters
        ----------
        model: torch.nn.Module
            the model to use for inference.
        Returns
        -------
        X_transformed: np.ndarray
            the transformed data.
        """

        if isinstance(self.X, np.ndarray):
            self.X = torch.from_numpy(self.X)
        num_samples = self.X.shape[0]
        n_batches = int(np.ceil(num_samples / self.batch_size))

        X_transformed_all = []
        model.to(self.device)
        with torch.no_grad():
            for i in range(n_batches):
                batch_idx = (
                    i * self.batch_size,
                    min((i + 1) * self.batch_size, num_samples),
                )
                X_batch = self.X[batch_idx[0] : batch_idx[1]]
                X_batch = X_batch.to(self.device)
                X_transformed = model(X_batch.float())
                if self.device.type == "cuda":
                    X_transformed = X_transformed.cpu()
                X_transformed_all.append(X_transformed.detach().numpy())

        X_transformed = np.concatenate(X_transformed_all, axis=0)
        return X_transformed

    def batch_inference(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs batched inference on the dataset.
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
        for batch in tqdm(self.dataloader):
            imgs = batch["img"]
            labels = batch["lab"]
            all_labels.append(labels)
            for img in imgs:
                img_transformed = model.fit_transform(img[0]).flatten()
                imgs_transformed.append(np.expand_dims(img_transformed, axis=0))
        X_transformed = np.concatenate(imgs_transformed)
        labels = np.concatenate(all_labels)
        return X_transformed, labels

    def xrv_clf_inference(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs batched inference with the TXRV Classifier on the dataset.
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
        for batch in tqdm(self.dataloader):
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

    def xrv_ae_inference(self, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs batched inference with the TXRV Autoencoder on the dataset.
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
        for batch in tqdm(self.dataloader):
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