import os
import sys
from tkinter import SEL_FIRST
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.append("..")

from drift_detection.baseline_models.temporal.pytorch.utils import get_temporal_model, get_device

from drift_detection.utils.drift_detector_utils import (
    get_args, 
    ContextMMDWrapper, 
    LKWrapper, 
    recurrent_neural_network, 
    convolutional_neural_network, 
    feed_forward_neural_network
)

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

    def __init__(self, test_method: str):
        self.tester_method = test_method

        #dict where the key is the string of each test_method and the value is the class of the test_method
        self.tester_methods = {
            "lk": LKWrapper,
            "lsdd": LSDDDrift,
            "mmd": MMDDrift,
            "tabular": TabularDrift,
            "ctx_mmd": ContextMMDWrapper,
            "chi2": ChiSquareDrift,
            "fet": FETDrift,
            "spot_the_diff": SpotTheDiffDrift,
            "ks": KSDrift,
        }

        if self.tester_method not in self.tester_methods.keys():
            raise ValueError("Test method not supported, must be one of: {}".format(self.tester_methods.keys()))

    def get_available_test_methods(self):
        return list(self.tester_methods.keys())
    
    def test_shift(self, X_s, X_t, **kwargs):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")
        
        self.method = self.tester_methods[self.tester_method](X_s, **get_args(self.tester_methods[self.tester_method], kwargs))

        preds = self.method.predict(X_t, **get_args(self.method.predict, kwargs))
    
        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist


class DCTester:
    """
    Domain Classifier Tester 

    Parameters
    ----------
    model: str
        model to use for domain classification. Must be one of: "gb", "rf", "rnn", "cnn", "ffnn"

    """

    def __init__(self, model_method: str, **kwargs):

        self.model_method = model_method

        self.model_methods = {
            "gb": GradientBoostingClassifier,
            "rf": RandomForestClassifier,
            "rnn": recurrent_neural_network,
            "cnn": convolutional_neural_network,
            "ffnn": feed_forward_neural_network,
        }

        if self.model_method not in self.model_methods.keys():
            raise ValueError("Model not supported, must be one of: {}".format(self.model_methods.keys()))
    
    def test_shift(self, X_s, X_t, **kwargs):
        X_s = X_s.astype("float32")
        X_t = X_t.astype("float32")

        self.model = self.model_methods[self.model_method](**get_args(self.model_methods[self.model_method], kwargs))

        preds = ClassifierDrift(X_s, self.model, **get_args(ClassifierDrift, kwargs)).predict(X_t)
        
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
    
