import os
import pandas as pd
import alibi
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import scipy
import scipy.stats as stats
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
from alibi_detect.cd.pytorch import preprocess_drift, HiddenOutput
from shift_utils import get_scaler

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
        
    def __init__(self, X, y, dr_tech, orig_dims, datset, dr_amount=None, var_ret=0.9,scaler="standard"):
        self.X = X
        self.y = y
        self.dr_tech = dr_tech
        self.orig_dims = orig_dims
        self.datset = datset
        self.mod_path = None
        self.scaler = get_scaler(scaler)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     

        if dr_amount is None:
            pca = PCA(n_components=var_ret, svd_solver='full')
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
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            ##implement load_model function
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path)
            return self.neural_network_classifier()
        elif self.dr_tech == "BBSDh_FFNN":
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path)
            return self.neural_network_classifier()
        else:
            return None
     
    def reduce(self, model, X , batch_size=32):
        if self.dr_tech == "PCA" or self.dr_tech == "SRP" or self.dr_tech =="kPCA" or self.dr_tech == "Isomap":
            self.scaler.fit(X)
            return model.transform(X)
        elif self.dr_tech == "NoRed":
            return X
        elif self.dr_tech == "BBSDs_FFNN":    
            pred = preprocess_drift(x=X.astype('float32'), model=model,device=self.device,batch_size=batch_size)
            return pred 
        elif self.dr_tech == "BBSDh_FFNN":    
            pred = preprocess_drift(x=X.astype('float32'), model=model,device=self.device,batch_size=batch_size)
            pred = np.argmax(pred, axis=1)
            return pred

    def sparse_random_projection(self):
        self.scaler(self.X)
        srp = SparseRandomProjection(n_components=self.dr_amount)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        self.scaler(self.X)
        pca = PCA(n_components=self.dr_amount)
        pca.fit(self.X)
        return pca

    def kernel_principal_components_anaylsis(self,kernel="rbf"):
        kpca = KernelPCA(n_components=self.dr_amount, kernel = kernel)
        kpca.fit(self.X)
        return kpca
        
    def manifold_isomap(self):
        self.scaler(self.X)
        isomap = Isomap(n_components=self.dr_amount)
        isomap.fit(self.X)
        return(isomap)
        
    def neural_network_classifier(self):   
        data_dim = self.X.shape[-1]
        if model is None:
            ffnn = nn.Sequential(
                nn.Linear(data_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 8),
                nn.SiLU(),
                nn.Linear(8, 1),
            ).to(self.device)
        return ffnn
