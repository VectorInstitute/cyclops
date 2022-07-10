import sys
import math
import random
import numpy as np
import torch
from scipy.special import softmax
import tensorflow as tf
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture

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
    ContextMMDDrift
)

from alibi_detect.utils.pytorch.kernels import DeepKernel

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

    def __init__(self, sign_level=0.05, mt=None, model_path=None, features=None):
        self.sign_level = sign_level
        self.mt = mt
        self.model_path = model_path
        self.device = get_device()
        self.features = features

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
    
    def context(self, x, context_type="prediction probabilities", model_name="lstm"):
        if context_type == "prediction probabilities":
            model = self.get_timeseries_model(model_name, x.shape[2])
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(x).to(self.device)).cpu().numpy()
            return softmax(logits)
        if context_type == "cluster membership":
            n_clusters = 2 
            gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=2022)
            gmm.fit(x_train)

    def test_shift(self, X_s, X_t, backend = "pytorch", C_s=None, C_t=None):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")

        p_val = None
        dist = None

        if self.mt == "MMD":
            dd = MMDDrift(X_s, backend=backend, n_permutations=1000, p_val=0.05)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "LSDD":
            dd = LSDDDrift(X_s, p_val=0.05, n_permutations=1000, backend=backend)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "LK":
            if self.model_path:
                proj = self.lstm(X_s.shape[2])
                proj.load_state_dict(torch.load(self.model_path))
            else:
                proj = nn.Sequential(
                    nn.Linear(X_s.shape[-1], 32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
                ).to(self.device)
            kernel = DeepKernel(proj, eps=0.01)
            dd = LearnedKernelDrift(
                X_s, kernel, backend=backend, p_val=0.05, epochs=10, batch_size=32
            )
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Classifier":
            if self.model_path:
                model = self.lstm(X_s.shape[2])
                model.load_state_dict(torch.load(self.model_path))
            else:
                model = nn.Sequential(
                    nn.Linear(X_s.shape[-1], 32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
                ).to(self.device)
            dd = ClassifierDrift(
                X_s, model, backend=backend, p_val=0.05, preds_type="logits"
            )
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
                epochs=10,
                batch_size=32
            )
            preds = dd.predict(X_t)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Context-Aware MMD":
            C_s = self.context(X_s)
            C_t = self.context(X_t)
            dd = ContextMMDDrift(X_s, C_s, backend=backend, n_permutations=1000, p_val=0.05)
            preds = dd.predict(X_t, C_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Univariate":
            ## change this to get appropriate column indexing for feature map
            feature_map = {f: None for f in list(self.features)}
            dd = TabularDrift(X_s, p_val=0.05, categories_per_feature=feature_map)
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

        return p_val, dist