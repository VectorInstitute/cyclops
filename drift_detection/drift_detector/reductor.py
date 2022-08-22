import os
import sys
from tkinter import Y
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
from alibi_detect.cd.pytorch import preprocess_drift
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

sys.path.append("..")
from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import get_temporal_model, get_device

np.set_printoptions(precision=5)

class Reductor:

    """
    The Reductor class is used to reduce the dimensionality of the data. 
    
    The reductor is initialized with a dimensionality reduction method. 
    The reductor can then be fit to the data and used to transform the data.

    Example:
    --------
    >>> from drift_detection.baseline_models.temporal.pytorch.reductor import Reductor
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> reductor = Reductor("PCA")
    >>> reductor.load_dataset(X)
    >>> reductor.fit()
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
            "BBSDs_untrained_FFNN"
            "BBSDh_untrained_FFNN"
            "BBSDs_untrained_CNN"
            "BBSDh_untrained_CNN"
            "BBSDs_untrained_LSTM"
            "BBSDh_untrained_LSTM"
            "BBSDs_trained_LSTM"
            "BBSDh_trained_LSTM"
            "BBSDs_txrv_CNN"
            "TAE_txrv_CNN"
    model_path: String
        The path to the model to use for dimensionality reduction.
        If None, A new model is fit.
    var_ret: float
        The percentage of variance to retain for {"PCA", "SRP", "kPCA", "Isomap"}.
    gmm_n_clusters: int
        The number of clusters to use for "GMM".
    random_state: int
        The global random seed.
    """

    def __init__(
        self,
        dr_method,
        model_path=None,
        var_ret=0.8,
        gmm_n_clusters=2,
        random_state=42
        ):

        self.dr_method = dr_method
        self.model_path = model_path
        self.var_ret = var_ret
        self.gmm_n_clusters = gmm_n_clusters
        self.device = get_device()
        
        self.random_state = random_state
        np.random.seed(self.random_state) # set global random seed for all methods
        
    def load_dataset(self, X, y=None):
        '''
        Loads an (X, y) numpy dataset from memory. 

        Parameters
        ----------
        X: numpy.matrix
            covariate data
        y: numpy.matrix
            optional targets/labels for the data

        Shape:
            X: (n_samples, n_features)
            y: (n_samples, n_targets)
        '''
        self.X = X
        self.y = y

    def load_image_dataset(self, dataset):
        '''
        Loads a pytorch map-style image dataset

        Parameters
        ----------
        dataset: String
            name of dataset to load
        '''
        self.dataset = dataset

    def load_model(self):
        '''
        Load model from path
        '''
        if (self.dr_method == "PCA" 
            or self.dr_method == "SRP" 
            or self.dr_method == "kPCA" 
            or self.dr_method == "Isomap"
            or self.dr_method == "GMM"):
            self.model = pickle.load(open(self.model_path, "rb"))
        elif (self.dr_method == "BBSDs_trained_LSTM"
            or self.dr_method == "BBSDh_trained_LSTM"):
            self.model.load_state_dict(torch.load(self.model_path))
        print("Model loaded from {}".format(self.model_path))

    def save_model(self, output_file):
        raise NotImplementedError()

    def fit(self):
        '''
        Fits the reductor to the data
        '''
        if (hasattr(self, "dataset")):
            raise ValueError("For batched dataset, use only transform argument for dimensionality reduction")

        if self.dr_method == "PCA":
            self.model = self.principal_components_anaylsis()
        elif self.dr_method == "SRP":
            self.model = self.sparse_random_projection()
        elif self.dr_method == "kPCA":
            self.model = self.kernel_principal_components_anaylsis()
        elif self.dr_method == "Isomap":
            self.model = self.manifold_isomap()
        elif self.dr_method == "GMM":
            self.model = self.gaussian_mixture_model()
        elif self.dr_method == "BBSDs_untrained_FFNN":
            self.model = self.feed_foward_neural_network()
        elif self.dr_method == "BBSDh_untrained_FFNN":
            self.model = self.feed_foward_neural_network()
        elif self.dr_method == "BBSDs_untrained_CNN":
            self.model = self.convolutional_neural_network()
        elif self.dr_method == "BBSDh_untrained_CNN":
            self.model = self.convolutional_neural_network()
        elif self.dr_method == "BBSDs_untrained_LSTM":
            self.model = self.recurrent_neural_network("lstm", self.X.shape[2])
        elif self.dr_method == "BBSDh_untrained_LSTM":
            self.model = self.recurrent_neural_network("lstm", self.X.shape[2])
        elif self.dr_method == "BBSDs_trained_LSTM":
            self.model = self.recurrent_neural_network("lstm",self.X.shape[2])
            self.model.load_state_dict(torch.load(self.model_path))
        elif self.dr_method == "BBSDh_trained_LSTM":
            self.model = self.recurrent_neural_network("lstm", self.X.shape[2])
            self.model.load_state_dict(torch.load(self.model_path))
        elif self.dr_method == "BBSDs_txrv_CNN":
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        elif self.dr_method == "TAE_txrv_CNN":
            self.model = xrv.autoencoders.ResNetAE(weights="101-elastic")
        else:
            raise ValueError("Invalid dimensionality reduction method, check available methods with Reductor.get_available_dr_methods()")

    def get_available_dr_methods(self):
        '''
        Returns a list of available dimensionality reduction methods

        Returns
        -------
        list
            list of available dimensionality reduction methods
        '''
        return ["PCA", "SRP", "kPCA", "Isomap", "BBSDs_untrained_FFNN", "BBSDh_untrained_FFNN", "BBSDs_untrained_CNN", "BBSDh_untrained_CNN", 
        "BBSDs_untrained_LSTM", "BBSDh_untrained_LSTM", "BBSDs_trained_LSTM", "BBSDh_trained_LSTM", "BBSDs_txrv_CNN", "TAE_txrv_CNN"]
        
    def get_dr_amount(self):
        '''
        Returns the number of components to be used to retain the variation specified.

        Returns
        -------
        int
            number of components.
        '''
        pca = PCA(n_components=self.var_ret, svd_solver="full")
        pca.fit(self.X)
        return(pca.n_components_)
    
    def transform(self, batch_size=32, num_workers=None):  
        '''Transforms the data using the chosen dimensionality reduction method

        Parameters
        ----------
        batch_size: int
            batch size for LSTM inference/pytorch dataloader. Default: 32
        num_workers: int
            number of workers for pytorch dataloader. If None, uses max number of cpus.

        Returns
        -------
        X_transformed: numpy.matrix
            transformed data
        '''
        if num_workers is None:
            num_workers = os.cpu_count()

        if (
            self.dr_method == "PCA"
            or self.dr_method == "SRP"
            or self.dr_method == "kPCA"
            or self.dr_method == "Isomap"
        ):
            if hasattr(self, "X"):
                X_transformed = self.model.transform(self.X)
            elif hasattr(self, "dataset"):
                X_transformed, self.y = self.batch_inference(self.model)
        elif self.dr_method == "NoRed":
            if hasattr(self, "X"):
                if self.y is not None:
                    return self.X, self.y
                else:
                    return self.X
            elif hasattr(self, "dataset"):
                return NotImplementedError("NoRed not implemented for image datasets")
        elif "BBSDs" in self.dr_method:
            if "xrv_clf" in self.dr_method:
                self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
                X_transformed, self.y = self.xrv_clf_inference(self.model)
            else:
                X_transformed = preprocess_drift(
                    x=self.X.astype("float32"),
                    model=self.model,
                    device=self.device,
                    batch_size=batch_size,
                    )
        elif "BBSDh" in self.dr_method:
            X_transformed = preprocess_drift(
                x=self.X.astype("float32"),
                model=self.model,
                device=self.device,
                batch_size=batch_size,
            )
            X_transformed = np.where(X_transformed > 0.5, 1, 0)
        elif self.dr_method == "GMM":           
            X_transformed = self.model.predict_proba(self.X)
        elif self.dr_method == "TAE_txrv_clf":
            if num_workers is None:
                num_workers = os.cpu_count()
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
            X_transformed, self.y = self.xrv_ae_inference(self.model)
        if self.y is not None:
            return X_transformed, self.y
        else:
            return X_transformed
        
    def sparse_random_projection(self):
        '''
        Fits a sparse random projection model.

        Returns
        -------
        sklearn.decomposition.SparseRandomProjection
            sparse random projection model.
        '''
        n_components = self.get_dr_amount()
        srp = SparseRandomProjection(n_components=n_components)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        '''
        Fits a principal components analysis model.

        Returns
        -------
        sklearn.decomposition.PCA
            principal components analysis model.
        '''
        n_components = self.get_dr_amount()
        pca = PCA(n_components=n_components)
        pca.fit(self.X)
        return pca

    def kernel_principal_components_anaylsis(self, kernel="rbf"):
        '''
        Fits a kernel principal components analysis model.

        Parameters
        ----------
        kernel: str
            kernel to be used. Default: "rbf"
        
        Returns
        -------
        sklearn.decomposition.KernelPCA
            kernel principal components analysis model.
        '''
        n_components = self.get_dr_amount()
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        kpca.fit(self.X)
        return kpca

    def manifold_isomap(self):
        '''
        Fits an isomap model.

        Returns
        -------
        sklearn.manifold.Isomap
            isomap model.
        '''
        n_components = self.get_dr_amount()
        isomap = Isomap(n_components=n_components)
        isomap.fit(self.X)
        return isomap

    def feed_foward_neural_network(self):
        '''
        Creates a feed forward neural network model.

        Returns
        -------
        model: torch.nn.Module
            feed forward neural network model.
        '''
        data_dim = self.X.shape[-1]
        ffnn = nn.Sequential(
                nn.Linear(data_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
        ).to(self.device).eval()
        return ffnn
    
    def convolutional_neural_network(self):
        '''
        Creates a convolutional neural network model.
        
        Returns
        -------
        torch.nn.Module
            convolutional neural network for dimensionality reduction.
        '''
        cnn = nn.Sequential(
                nn.Conv2d(3, 8, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(8, 16, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
        ).to(self.device).eval()
        return cnn
        
    def recurrent_neural_network(self, model_name, input_dim, hidden_dim = 64, layer_dim = 2, dropout = 0.2, output_dim = 1, last_timestep_only = False):
        '''
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
        '''
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
        '''
        Creates a gaussian mixture model.
        '''
        gmm = GaussianMixture(n_components=self.gmm_n_clusters, covariance_type='full')
        gmm.fit(self.X)
        return gmm

    def batch_inference(self, model):
        '''
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
        '''
        imgs_transformed = []
        all_labels = []
        for batch in tqdm(self.dataloader):
            imgs = batch['img']
            labels = batch['lab']
            all_labels.append(labels)
            for img in imgs:
                imgs_transformed.append(model.fit_transform(img))
        X_transformed = np.concatenate(imgs_transformed)
        labels = np.concatenate(all_labels)
        return X_transformed, labels
    
    def xrv_clf_inference(self, model):
        '''
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
        '''
        all_preds = []
        all_labels = []
        model = model.to(self.device).eval()
        for batch in tqdm(self.dataloader):
            imgs = batch['img']
            labels = batch['lab']
            imgs = torch.from_numpy(imgs).to(self.device)
            with torch.no_grad():
                preds = model(imgs)
            preds = preds.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)
        X_transformed = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return X_transformed, labels

    def xrv_ae_inference(self, model):
        '''
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
        '''
        all_preds = []
        all_labels = []
        model = model.to(self.device).eval()
        for batch in tqdm(self.dataloader):
            imgs = batch['img']
            labels = batch['lab']
            imgs = torch.from_numpy(imgs).to(self.device)
            with torch.no_grad():
                preds = model.encode(imgs).mean(dim=(2,3))
            preds = preds.cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels)        
        X_transformed = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        return X_transformed, labels
            

