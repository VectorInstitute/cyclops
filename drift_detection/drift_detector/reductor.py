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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.random_projection import SparseRandomProjection
from alibi_detect.cd.pytorch import HiddenOutput, preprocess_drift
#from pyts.decomposition import SingularSpectrumAnalysis

sys.path.append("..")

from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import *

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
    orig_dims: int
        number of dimensions of data
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
        orig_dims,
        datset,
        var_ret=0.8,
        model_path=None):
        
        self.X = X
        self.y = y
        self.dr_tech = dr_tech
        self.orig_dims = orig_dims
        self.datset = datset
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
            return self.neural_network_classifier()
        elif self.dr_tech == "BBSDh_untrained_FFNN":
            return self.neural_network_classifier()
        elif self.dr_tech == "BBSDs_untrained_LSTM":
            return self.get_timeseries_model("lstm", self.X.shape[2])
        elif self.dr_tech == "BBSDs_trained_LSTM":
            model = self.get_timeseries_model("lstm",self.X.shape[2])
            model.load_state_dict(torch.load(self.model_path))
            return model
        elif self.dr_tech == "BBSDh_trained_LSTM":
            model = self.get_timeseries_model("lstm", self.X.shape[2])
            model.load_state_dict(torch.load(self.model_path))
            return model
        else:
            return None

    def reduce(self, model, X, batch_size=32):          
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
            pred = np.argmax(pred, axis=1)
            return pred  

    def sparse_random_projection(self):
        n_components = self.get_dr_amount()
        srp = SparseRandomProjection(n_components=self.dr_amount)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        n_components = self.get_dr_amount()
        pca = PCA(n_components=self.dr_amount)
        pca.fit(self.X)
        return pca

    def kernel_principal_components_anaylsis(self, kernel="rbf"):
        n_components = self.get_dr_amount()
        kpca = KernelPCA(n_components=self.dr_amount, kernel=kernel)
        kpca.fit(self.X)
        return kpca

    def manifold_isomap(self):
        n_components = self.get_dr_amount()
        isomap = Isomap(n_components=self.dr_amount)
        isomap.fit(self.X)
        return isomap

    def neural_network_classifier(self):
        data_dim = self.X.shape[-1]
        ffnn = nn.Sequential(
                nn.Linear(data_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
        ).to(self.device)
        return ffnn
    
    def get_timeseries_model(self, model_name, input_dim, hidden_dim = 64, layer_dim = 2, dropout = 0.2, output_dim = 1, last_timestep_only = False):
        model_params = {
            "device": self.device,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layer_dim": layer_dim,
            "output_dim": output_dim,
            "dropout_prob": dropout,
            "last_timestep_only": last_timestep_only,
        }
        model = get_temporal_model(model_name, model_params).to(self.device)
        return model
