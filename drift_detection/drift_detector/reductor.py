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
        dr_amount=None,
        var_ret=0.9,
        scale=False,
        scaler="standard",
        model=None):
        
        self.X = X
        self.y = y
        self.dr_tech = dr_tech
        self.orig_dims = orig_dims
        self.datset = datset
        self.model = model
        self.scale = scale
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if scale:
            self.scaler = self.get_scaler(scaler)
            self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)

        if dr_amount is None:
            pca = PCA(n_components=var_ret, svd_solver="full")
            pca.fit(X)
            self.dr_amount = pca.n_components_
        else:
            self.dr_amount = dr_amount

    def fit_reductor(self):    
        if self.dr_tech == "PCA":
            return self.principal_components_anaylsis()
        elif self.dr_tech == "SRP":
            return self.sparse_random_projection()
        elif self.dr_tech == "kPCA":
            return self.kernel_principal_components_anaylsis()
        elif self.dr_tech == "Isomap":
            return self.manifold_isomap()
        elif self.dr_tech == "BBSDs_FFNN":
            if self.model:
                return self.model
            return self.neural_network_classifier()
        elif self.dr_tech == "BBSDh_FFNN":
            if self.model:
                return self.model
            return self.neural_network_classifier()
        elif self.dr_tech == "BBSDs_LSTM":
            if self.model:
                return self.model
            return self.lstm()
        else:
            return None
        
    def get_scaler(self, scaler):
        """Get scaler.

        Parameters
        ----------
        scaler: string
            String indicating which scaler to retrieve.

        """
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()


    def reduce(self, model, X, batch_size=32):
        if self.scale:
            X = self.scaler.transform(X)
            
        if (
            self.dr_tech == "PCA"
            or self.dr_tech == "SRP"
            or self.dr_tech == "kPCA"
            or self.dr_tech == "Isomap"
        ):
            return model.transform(X)
        elif self.dr_tech == "NoRed":
            return X
        elif self.dr_tech == "BBSDs_FFNN":
            pred = preprocess_drift(
                x=X.astype("float32"),
                model=model,
                device=self.device,
                batch_size=batch_size,
            )
            return pred
        elif self.dr_tech == "BBSDh_FFNN":
            pred = preprocess_drift(
                x=X.astype("float32"),
                model=model,
                device=self.device,
                batch_size=batch_size,
            )
            pred = np.argmax(pred, axis=1)
            return pred

    def sparse_random_projection(self):
        srp = SparseRandomProjection(n_components=self.dr_amount)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        pca = PCA(n_components=self.dr_amount)
        pca.fit(self.X)
        return pca

    def kernel_principal_components_anaylsis(self, kernel="rbf"):
        kpca = KernelPCA(n_components=self.dr_amount, kernel=kernel)
        kpca.fit(self.X)
        return kpca

    def manifold_isomap(self):
        isomap = Isomap(n_components=self.dr_amount)
        isomap.fit(self.X)
        return isomap

    def neural_network_classifier(self):
        data_dim = self.X.shape[-1]
        if not self.model:
            ffnn = nn.Sequential(
                nn.Linear(data_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
            ).to(self.device)
        return ffnn
    
    def lstm(self):
        raise NotImplementedError
