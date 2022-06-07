import math
import random
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from scipy.spatial import distance

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

    def __init__(self, sign_level=0.05, mt=None, mod_path=None):
        self.sign_level = sign_level
        self.mt = mt
        self.mod_path = mod_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_shift(self, X_s, X_t, backend = "pytorch"):
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
            dd = LSDDDrift(X_s, p_val=0.05, backend=backend)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "LK":
            if self.mod_path:
                proj = load_model(self.mod_path)
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
            if self.mod_path:
                model = load_model(self.mod_path)
            else:
                model = nn.Sequential(
                    nn.Linear(X_s.shape[-1], 32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 2),
                ).to(self.device)
            dd = ClassifierDrift(
                X_s, model, backend=backend, p_val=0.05, preds_type="logits"
            )
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.mt == "Univariate":
            ## change this to get appropriate column indexing for feature map
            feature_map = {f: None for f in list(X_s.columns)}
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