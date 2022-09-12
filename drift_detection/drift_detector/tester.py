import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from alibi_detect.utils.pytorch.kernels import DeepKernel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.append("..")

from drift_detection.baseline_models.temporal.pytorch.utils import get_temporal_model, get_device

from alibi_detect.cd import (
    ClassifierDrift,
    LearnedKernelDrift,
    LSDDDrift,
    MMDDrift,
    TabularDrift,
    ContextMMDDrift,
    ChiSquareDrift,
    FETDrift,
    SpotTheDiffDrift,
    KSDrift,
)

class TSTester:

    """
    Two Sample Statistical Test Methods 

    Parameters
    ----------
    sign_level: float
        significance level

    """

    def __init__(self, test_method, model_path=None, lk_projection=None, context_type=None):
        self.test_method = test_method
        self.model_path = model_path
        self.lk_projection = lk_projection
        self.context_type = context_type
        self.device = get_device()
    
    def test_shift(self, X_s, X_t):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")

        if self.test_method == "MMD":
            dd = MMDDrift(X_s, backend=backend, n_permutations=100)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)

        if self.test_method == "KS":
            dd = KSDrift(X_s, backend=backend, n_permutations=100)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)

        elif self.test_method == "LSDD":
            dd = LSDDDrift(X_s, n_permutations=100, backend=backend)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)

        elif self.test_method == "LK":
            
            if self.lk_projection == "rnn":
                proj = self.recurrent_neural_network("lstm",X_s.shape[2])
                if self.model_path is not None:   
                    proj.load_state_dict(torch.load(self.model_path))
                    
            elif self.lk_projection == "cnn":             
                # define the projection phi
                proj = nn.Sequential(
                    nn.LazyConv2d(8, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                ).to(self.device).eval()
                
            elif self.lk_projection == "ffnn":
                proj = nn.Sequential(
                    nn.LazyLinear(32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
                ).to(self.device).eval()
            
            else:
                raise ValueError("Unspecified kernel projection type, please choose from 'rnn', 'cnn', 'ffnn'")
                
            kernel = DeepKernel(proj, eps=0.01)
            
            dd = LearnedKernelDrift(
                X_s, kernel, backend=backend, p_val=0.05, epochs=100, batch_size=32
            )
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)

        elif self.test_method == "Spot-the-diff":
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

        elif self.test_method == "Context-Aware MMD":
            C_s = self.context(X_s, self.context_type)
            C_t = self.context(X_t, self.context_type)
            dd = ContextMMDDrift(X_s, C_s, backend=backend, n_permutations=100, p_val=0.05)
            preds = dd.predict(X_t, C_t, return_p_val=True, return_distance=True)

        elif self.test_method == "Univariate":
            X_s = X_s.reshape(X_s.shape[0],X_s.shape[1])
            X_t = X_t.reshape(X_t.shape[0],X_t.shape[1])
            ## add feature map
            dd = TabularDrift(X_s, correction = "bonferroni", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            
        elif self.test_method == "Chi-Squared":
            dd = ChiSquareDrift(X_s, correction = "bonferroni", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            
        elif self.test_method == "FET":
            dd = FETDrift(X_s, alternative="two-sided", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
        
        else:
            raise ValueError("Incorrect Representation Option")

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist


class DCTester:
    """
    Domain Classifier Tester 

    Parameters
    ----------
    sign_level: float
        significance level

    """

    def __init__(self, model, sign_level=0.05, model_path=None):

        self.model = model
        self.sign_level = sign_level
        self.model_path = model_path
        self.device = get_device()


    
    def test_shift(self, X_s, X_t, **kwargs):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")

        p_val = None
        dist = None

        # check model type and select backend:
        model_type = str(type(self.model))[8:].split('.')[0]
        if model_type == 'sklearn':
            backend = 'pytorch'
        elif model_type == 'torch':
            backend = 'pytorch'
        else:
            raise ValueError("Incorrect model type, backend must be 'pytorch' or 'sklearn', not {}".format(model_type))

        if self.model == "gb":
            model = GradientBoostingClassifier()
            preds = ClassifierDrift(
                X_s, 
                model, 
                backend=backend, 
                preds_type='scores', 
                **kwargs
            )
        
        elif self.model == "rf":
            model = RandomForestClassifier()
            preds = ClassifierDrift(
                X_s, 
                model, 
                backend=backend, 
                **kwargs
            )       
        
        elif self.model == "rnn":
            model = self.recurrent_neural_network("lstm", X_s.shape[-1])
            preds = ClassifierDrift(
                X_s, 
                model, 
                backend=backend,
                **kwargs
            )   
        
        elif self.model == "ffnn":
            model = nn.Sequential(
                    nn.LazyLinear(32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
            ).to(self.device)
            preds = ClassifierDrift(
                X_s, 
                model, 
                backend=backend,
                **kwargs
            ) 
        
        else:
            raise ValueError("Incorrect Representation Option")
        
        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist


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

    def __init__(self, sign_level=0.05, mt=None, model_path=None,):
        self.sign_level = sign_level
        self.test_method = mt
        self.model_path = model_path
        self.device = get_device()
    
    def test_shift(self, X_s, X_t, context_type, representation = None, backend = "pytorch"):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")

        p_val = None
        dist = None

        if self.test_method == "MMD":
            dd = MMDDrift(X_s, backend=backend, n_permutations=100, p_val=0.05)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.test_method == "LSDD":
            dd = LSDDDrift(X_s, p_val=0.05, n_permutations=100, backend=backend)
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.test_method == "LK":
            if representation == "rnn":
                proj = self.recurrent_neural_network("lstm",X_s.shape[2])
                if self.model_path is not None:   
                    proj.load_state_dict(torch.load(self.model_path))
                    
            elif representation == "cnn":             
                # define the projection phi
                proj = nn.Sequential(
                    nn.LazyConv2d(8, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                ).to(self.device).eval()
                
            elif representation == "ffnn":
                proj = nn.Sequential(
                    nn.LazyLinear(32),
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

        elif self.test_method == "Spot-the-diff":
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

        elif self.test_method == "Context-Aware MMD":
            C_s = self.context(X_s, context_type)
            C_t = self.context(X_t, context_type)
            dd = ContextMMDDrift(X_s, C_s, backend=backend, n_permutations=100, p_val=0.05)
            preds = dd.predict(X_t, C_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]

        elif self.test_method == "Univariate":
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
            
        elif self.test_method == "Chi-Squared":
            dd = ChiSquareDrift(X_s, correction = "bonferroni", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]
            
        elif self.test_method == "FET":
            dd = FETDrift(X_s, alternative="two-sided", p_val=0.05)
            preds = dd.predict(
                X_t, drift_type="batch", return_p_val=True, return_distance=True
            )
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]
            
        else:
            raise ValueError("Incorrect Representation Option")
                
        return p_val, dist


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
    
    def context(self, x, context_type="rnn", n_clusters=2):
        if context_type == "rnn":
            model = self.recurrent_neural_network("lstm", x.shape[2])
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
                return self.context(x, "rnn")
        
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
    
