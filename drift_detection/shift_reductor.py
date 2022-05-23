import os
import pandas as pd
import alibi
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import scipy
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
from alibi_detect.cd.pytorch import preprocess_drift, HiddenOutput
import scipy.stats as stats

class ShiftReductor:

    """ShiftReductor Class.

    Attributes
    ----------
    X: numpy.matrix
        covariate data for source
    y: list
        labels for source
    X_t: numpy.matrix
        covariate data for target
    y_t: list
        labels for target
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
        
    def __init__(self, X, y, X_t, y_t, dr_tech, orig_dims, datset, dr_amount=None, var_ret=0.9):
        self.X = X
        self.y = y
        self.X_t = X_t
        self.y_t = y_t
        self.dr_tech = dr_tech
        self.orig_dims = orig_dims
        self.datset = datset
        self.mod_path = None
        self.scaler = StandardScaler()
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
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path, custom_objects=keras_resnet.custom_objects)
            return self.neural_network_classifier()
        elif self.dr_tech == "BBSDh_FFNN":
            self.mod_path = './saved_models/' + self.datset + '_standard_class_model.h5'
            if os.path.exists(self.mod_path):
                return load_model(self.mod_path, custom_objects=keras_resnet.custom_objects)
            return self.neural_network_classifier()
        else:
            return None
     
    def reduce(self, model, X):
        if self.dr_tech == "PCA" or self.dr_tech == "SRP" or self.dr_tech =="kPCA" or self.dr_tech == "Isomap":
            self.scaler.fit(X)
            return model.transform(X)
        elif self.dr_tech == "NoRed":
            return X
        elif self.dr_tech == "BBSDs_FFNN":    
            pred = preprocess_drift(x=X.astype('float32'), model=model,device=self.device,batch_size=32)
            return pred 
        elif self.dr_tech == "BBSDh_FFNN":    
            pred = preprocess_drift(x=X.astype('float32'), model=model,device=self.device,batch_size=32)
            pred = np.argmax(pred, axis=1)
            return pred

    def sparse_random_projection(self):
        srp = SparseRandomProjection(n_components=self.dr_amount)
        srp.fit(self.X)
        return srp

    def principal_components_anaylsis(self):
        self.scaler.fit(self.X)
        pca = PCA(n_components=self.dr_amount)
        pca.fit(self.X)
        return pca

    def kernel_principal_components_anaylsis(self):
        self.scaler.fit(self.X)
        kpca = KernelPCA(n_components=self.dr_amount, kernel = 'rbf')
        kpca.fit(self.X)
        return kpca
        
    def manifold_isomap(self):
        self.scaler.fit(self.X)
        isomap = Isomap(n_components=self.dr_amount)
        isomap.fit(self.X)
        return(isomap)
        
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
