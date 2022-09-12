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

from alibi_detect.cd import (
    ClassifierDrift,
    SpotTheDiffDrift
)

class ShiftClassifier:

    """ShiftClassifier Class.
    Attributes
    ----------
    sign_level: float
        P-value significance level.
    mt: String
        Name of classifier method.
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
    
    def feed_forward_neural_network(self, input_dim):
        model = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.SiLU(),
                    nn.Linear(32, 8),
                    nn.SiLU(),
                    nn.Linear(8, 1),
                ).to(self.device)
        return model
    
    def test_shift(self, X_s, X_t, classifier_type = "ffnn" , backend = "pytorch"):
        X_s = X_s.astype(np.float32)
        X_t = X_t.astype(np.float32)

        p_val = None
        dist = None

        if self.mt == "Classifier":
            if classifier_type == "gb":
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
            elif classifier_type == "rf":
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
            elif classifier_type == "rnn":
                model = self.recurrent_neural_network("lstm",X_s.shape[2])
                backend='pytorch'
                dd = ClassifierDrift(
                    X_s, 
                    model, 
                    backend=backend, 
                    p_val=0.05
                ) 
                
            elif classifier_type == "ffnn":
                model = self.feed_forward_neural_network(X_s.shape[-1])
                dd = ClassifierDrift(
                    X_s, 
                    model, 
                    backend=backend, 
                    p_val=0.05
                ) 
            else:
                raise ValueError("Incorrect Classifier Type")
                
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
                batch_size=1,
            )
            preds = dd.predict(X_t, return_p_val=True, return_distance=True)
            p_val = preds["data"]["p_val"]
            dist = preds["data"]["distance"]
            
        else:
            raise ValueError("Incorrect Classifier Method")
                
        return p_val, dist
    
