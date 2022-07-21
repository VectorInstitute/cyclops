import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from scipy.special import softmax
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from alibi_detect.utils.pytorch.kernels import DeepKernel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.append("..")

from drift_detection.baseline_models.temporal.pytorch.optimizer import Optimizer
from drift_detection.baseline_models.temporal.pytorch.utils import *

from scipy.stats import (
    anderson_ksamp,
    binom_test,
    chi2_contingency,
    chisquare,
    ks_2samp,
)

from alibi_detect.cd import (
    ClassifierDrift,
    LearnedKernelDrift,
    LSDDDrift,
    LSDDDriftOnline,
    MMDDrift,
    MMDDriftOnline,
    TabularDrift,
    ContextMMDDrift,
    ChiSquareDrift,
    FETDrift,
    SpotTheDiffDrift
)

class ShiftTester:

    """ShiftTester Class.
    Attributes
    ----------
    sign_level: float
        P-value significance level.
    mt: String
        Name of two sample hypothesis test.
    mod_path: String
        path to model (optional)
    """

    def __init__(self, sign_level=0.05, mt=None, model_path=None, features=None, dataset=None):
        self.sign_level = sign_level
        self.mt = mt
        self.model_path = model_path
        self.device = get_device()
        self.features = features
        self.dataset = dataset

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
        model = get_temporal_model(model_name, model_params).to(self.device)
        return model
    
    def context(self, x, context_type="lstm", n_clusters=2):
        if context_type == "lstm":
            model = self.recurrent_neural_network(context_type, x.shape[2])
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(self.device)).cpu().numpy()
            return softmax(logits, -1)
        if context_type == "gmm":
            gmm = self.gaussian_mixture_model(n_clusters)
            if gmm is not None:
                # compute all contexts
                c_gmm_proba = gmm.predict_proba(x) 
                return c_gmm_proba
            else: 
                return self.context(x, "lstm")
        
    def gaussian_mixture_model(self, n_clusters=2):
        gmm=None
        if os.path.exists(self.dataset + '_' + str(n_clusters) + '_means.npy'):
            n_clusters=str(n_clusters)
            means = np.load(self.dataset + '_' + n_clusters + '_means.npy')
            covar = np.load(self.dataset + '_' + n_clusters + '_covariances.npy')
            gmm = GaussianMixture(n_components = len(means), covariance_type='full')
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
            gmm.weights_ = np.load(self.dataset + '_' + n_clusters + '_weights.npy')
            gmm.means_ = means
            gmm.covariances_ = covar
        return gmm
    
    def test_shift(self, X_s, X_t, context_type, representation = None, backend = "pytorch"):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")

        p_val = None
        dist = None

        if self.mt == "MMD":
            dd = MMDDrift(X_s, backend=backend, n_permutations=100, p_val=0.05)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "LSDD":
            dd = LSDDDrift(X_s, p_val=0.05, n_permutations=100, backend=backend)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "LK":
            if representation == "lstm":
                proj = self.lstm(X_s.shape[2])
                if self.model_path is not None:   
                    proj.load_state_dict(torch.load(self.model_path))
                    
            elif representation == "cnn":             
                # define the projection phi
                proj = nn.Sequential(
                    nn.Conv2d(3, 8, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                ).to(self.device).eval()
                
            elif representation == "ffnn":
                proj = nn.Sequential(
                    nn.Linear(X_s.shape[-1], 32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
                ).to(self.device).eval()
            else:
                raise ValueError("Incorrect Representation Option")
                
            kernel = DeepKernel(proj, eps=0.01)
            
            dd = LearnedKernelDrift(
                X_s, kernel, backend=backend, p_val=0.05, epochs=100, batch_size=32
            )
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Classifier":
            if representation == "gb":
                X_s = X_s.reshape(X_s.shape[0],X_s.shape[1])
                X_t = X_t.reshape(X_t.shape[0],X_t.shape[1])
                model = GradientBoostingClassifier()
                backend='sklearn'
                dd = ClassifierDrift(
                    X_s, 
                    model, 
                    backend=backend, 
                    p_val=0.05, 
                    preds_type='scores', 
                    binarize_preds=False,
                    n_folds=5
                )        
            elif representation == "rf":
                X_s = X_s.reshape(X_s.shape[0],X_s.shape[1])
                X_t = X_t.reshape(X_t.shape[0],X_t.shape[1])
                model = RandomForestClassifier()
                backend='sklearn'
                dd = ClassifierDrift(
                    X_s, 
                    model, 
                    backend=backend, 
                    p_val=0.05, 
                    binarize_preds=False,
                    n_folds=5
                )             
            elif representation == "lstm":
                model = self.lstm(X_s.shape[2])
                backend='pytorch'
                dd = ClassifierDrift(
                    X_s, 
                    model, 
                    backend=backend, 
                    p_val=0.05
                ) 
                
            elif representation == "ffnn":
                model = nn.Sequential(
                        nn.Linear(X_s.shape[-1], 32),
                        nn.SiLU(),
                        nn.Linear(32, 8),
                        nn.SiLU(),
                        nn.Linear(8, 1),
                ).to(self.device)
                dd = ClassifierDrift(
                    X_s, 
                    model, 
                    backend=backend, 
                    p_val=0.05
                ) 
            else:
                raise ValueError("Incorrect Representation Option")
                
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Spot-the-diff":
            dd = SpotTheDiffDrift(
                X_s,
                backend=backend,
                p_val=.05,
                n_diffs=1,
                l1_reg=1e-3,
                epochs=100,
                batch_size=1
            )
            preds = dd.predict(X_t)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Context-Aware MMD":
            C_s = self.context(X_s, context_type)
            C_t = self.context(X_t, context_type)
            dd = ContextMMDDrift(X_s, C_s, backend=backend, n_permutations=100, p_val=0.05)
            preds = dd.predict(X_t, C_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Univariate":
            X_s = X_s.reshape(X_s.shape[0],X_s.shape[1])
            X_t = X_t.reshape(X_t.shape[0],X_t.shape[1])
            ## add feature map
            dd = TabularDrift(X_s, correction = "bonferroni", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            p_val = min(np.min(preds["data"]["p_val"]),1.0)
            dist = np.mean(preds["data"]["distance"])
            threshold = preds['data']['threshold']
            
        elif self.mt == "Chi-Squared":
            dd = ChiSquareDrift(X_s, correction = "bonferroni", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]
            
        elif self.mt == "FET":
            dd = FETDrift(X_s, alternative="two-sided", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        #need torch two sample for below
        elif self.mt == "Energy":
            energy_test = EnergyStatistic(len(X_s), len(X_t))
            t_val, matrix = energy_test(
                torch.autograd.Variable(torch.tensor(X_s)),
                torch.autograd.Variable(torch.tensor(X_t)),
                ret_matrix=True,
            )
            p_val = energy_test.pval(matrix)

        elif self.mt == "FR":
            fr_test = FRStatistic(len(X_s), len(X_t))
            t_val, matrix = fr_test(
                torch.autograd.Variable(torch.tensor(X_s)),
                torch.autograd.Variable(torch.tensor(X_t)),
                norm=2,
                ret_matrix=True,
            )
            p_val = fr_test.pval(matrix)

        elif self.mt == "KNN":
            knn_test = KNNStatistic(len(X_s), len(X_t), 20)
            t_val, matrix = knn_test(
                torch.autograd.Variable(torch.tensor(X_s)),
                torch.autograd.Variable(torch.tensor(X_t)),
                norm=2,
                ret_matrix=True,
            )
            p_val = knn_test.pval(matrix)
            
        else:
            raise ValueError("Incorrect Representation Option")
                
        return p_val, dist
    
