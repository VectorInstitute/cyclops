import os
import sys
import alibi
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import tensorflow as tf
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
from alibi_detect.cd.pytorch import HiddenOutput, preprocess_drift
#from pyts.decomposition import SingularSpectrumAnalysis

sys.path.append("..")

from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import *

np.set_printoptions(precision=5)

class ShiftReductor:

    """ShiftReductor Class.

    Attributes
    ----------
    X: numpy.matrix
        covariate data for source
    y: list
        labels for source
    dr_tech: String
        dimensionality reduction technique (e.g. PCA, BBSDs)
    datset: String
        dataset name
    dr_amount: int
        number of components to reduce pca to, defaults to all components
    var_ret: float
        variation retained for pca, defaults to 0.9

    """

    def __init__(
        self,
        X,
        y,
        dr_tech,
        datset,
        var_ret=0.8,
        model_path=None):
        
        self.X = X
        self.y = y
        self.dr_tech = dr_tech
        self.dataset = datset
        self.model_path = model_path
        self.var_ret = var_ret
        self.device = get_device()
        
    def get_dr_amount(self):
        pca = PCA(n_components=self.var_ret, svd_solver="full")
        pca.fit(self.X)
        return(pca.n_components_)

    def fit_reductor(self):    
        if self.dr_tech == "PCA":
            return self.principal_components_anaylsis()
        elif self.dr_tech == "SRP":
            return self.sparse_random_projection()
        elif self.dr_tech == "kPCA":
            return self.kernel_principal_components_anaylsis()
        elif self.dr_tech == "Isomap":
            return self.manifold_isomap()
        elif self.dr_tech == "BBSDs_untrained_FFNN":
            return self.feed_foward_neural_network()
        elif self.dr_tech == "BBSDh_untrained_FFNN":
            return self.feed_foward_neural_network()
        elif self.dr_tech == "BBSDs_untrained_CNN":
            return self.convolutional_neural_network()
        elif self.dr_tech == "BBSDh_untrained_CNN":
            return self.convolutional_neural_network()
        elif self.dr_tech == "BBSDs_untrained_LSTM":
            return self.recurrent_neural_network("lstm", self.X.shape[2])
        elif self.dr_tech == "BBSDh_untrained_LSTM":
            return self.recurrent_neural_network("lstm", self.X.shape[2])
        elif self.dr_tech == "BBSDs_trained_LSTM":
            model = self.recurrent_neural_network("lstm",self.X.shape[2])
            model.load_state_dict(torch.load(self.model_path))
            return model
        elif self.dr_tech == "BBSDh_trained_LSTM":
            model = self.recurrent_neural_network("lstm", self.X.shape[2])
            model.load_state_dict(torch.load(self.model_path))
            return model
        else:
            return None

    def reduce(self, model, X, batch_size=1, n_clusters=2):          
        if (
            self.dr_tech == "PCA"
            or self.dr_tech == "SRP"
            or self.dr_tech == "kPCA"
            or self.dr_tech == "Isomap"
        ):
            return model.transform(X)
        elif self.dr_tech == "NoRed":
            return X
        elif "BBSDs" in self.dr_tech:
            pred = preprocess_drift(
                x=X.astype("float32"),
                model=model,
                device=self.device,
                batch_size=batch_size,
            )
            return pred
        elif "BBSDh" in self.dr_tech:
            pred = preprocess_drift(
                x=X.astype("float32"),
                model=model,
                device=self.device,
                batch_size=batch_size,
            )
            pred = np.where(pred > 0.5, 1, 0)
            return pred  
        elif self.dr_tech == "GMM":           
            gmm = self.gaussian_mixture_model(n_clusters)
            # compute all contexts
            return gmm.predict_proba(X) 

    def sparse_random_projection(self):
        n_components = self.get_dr_amount()
        srp = SparseRandomProjection(n_components=n_components)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        n_components = self.get_dr_amount()
        pca = PCA(n_components=n_components)
        pca.fit(self.X)
        return pca

    def kernel_principal_components_anaylsis(self, kernel="rbf"):
        n_components = self.get_dr_amount()
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        kpca.fit(self.X)
        return kpca

    def manifold_isomap(self):
        n_components = self.get_dr_amount()
        isomap = Isomap(n_components=n_components)
        isomap.fit(self.X)
        return isomap

    def feed_foward_neural_network(self):
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

    def gaussian_mixture_model(self, n_clusters=2):
        if os.path.exists(self.dataset + '_' + str(n_clusters) + '_means.npy'):
            n_clusters=str(n_clusters)
            means = np.load(self.dataset + '_' + n_clusters + '_means.npy')
            covar = np.load(self.dataset + '_' + n_clusters + '_covariances.npy')
            gmm = GaussianMixture(n_components = len(means), covariance_type='full')
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
            gmm.weights_ = np.load(self.dataset + '_' + n_clusters + '_weights.npy')
            gmm.means_ = means
            gmm.covariances_ = covar  
        else:
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=2022)
            gmm.fit(self.X)
            n_clusters=str(n_clusters)
            np.save(self.dataset + '_' + n_clusters + '_weights', gmm.weights_, allow_pickle=False)
            np.save(self.dataset + '_' + n_clusters + '_means', gmm.means_, allow_pickle=False)
            np.save(self.dataset + '_' + n_clusters + '_covariances', gmm.covariances_, allow_pickle=False)
        return gmm
        