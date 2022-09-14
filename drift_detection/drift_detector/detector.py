from drift_detector import Reductor, TSTester, DCTester
from typing import List, Tuple, Union, Optional, Callable, Any, Dict

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
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import SparseRandomProjection
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from drift_detection.utils.drift_detection_utils import get_args

class Detector:

    """
    Detector class for distribution shift detection. 


    Attributes
    ----------
    reductor : str or Reductor
        Reductor object for dimensionality reduction.
    tester : str or TSTester or DCTester
        Tester object for statistical testing.
    p_val_threshold : float
        Threshold for p-value. If p-value is below this threshold, a shift is detected.
    

    Methods
    -------
    detect_shift(source_data, target_data, **kwargs)
        Detects shift between source and target data.
    """
    def __init__(
        self,
        reductor: Union[str, Reductor],
        tester: Union[str, TSTester, DCTester],
        p_val_threshold=0.05,
    ):

        self.reductor = reductor
        self.tester = tester
        self.p_val_threshold = p_val_threshold

    def _fit(self, data: Union[np.ndarray, torch.utils.data.Dataset]):
        self.reductor.fit(data)

    def _transform(self, X, **kwargs):
        return self.reductor.transform(X, **kwargs)
        
    def _test_shift(self, X_s, X_t, **kwargs):
        """Test for shift.
        """
        p_val, dist = self.tester.test_shift(X_s, X_t, **kwargs)

        return {'p_val': p_val, 'distance': dist}

    def detect_shift(self, source_data: Union[np.ndarray, torch.utils.data.Dataset],
                           target_data: Union[np.ndarray, torch.utils.data.Dataset], 
                           **kwargs):
        """Detects shift between source and target data.

        
        Parameters
        ----------
        source_data : np.ndarray or torch.utils.data.Dataset
            Source data.
        target_data : np.ndarray or torch.utils.data.Dataset
            Target data.
        **kwargs
            Keyword arguments for Reductor and TSTester.
        
        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.
        """
        
        self._fit(source_data)
        X_s = self._transform(source_data, **get_args(self.reductor.transform, kwargs))
        X_t = self._transform(target_data, **get_args(self.reductor.transform, kwargs))

        results = self._test_shift(X_s, X_t, **get_args(self.tester.test_shift, kwargs))

        if results['p_val'] < self.p_val_threshold:
            shift_detected = True
        else:
            shift_detected = False
        
        return {'p_val': results['p_val'], 'distance': results['distance'], 'shift_detected': shift_detected}

        # if self.reductor.dr_method != "BBSDh":
        #     # Lower the significance level for all tests (Bonferroni) besides BBSDh, which needs no correction.
        #     self.p_val = self.p_val / X_s_red.shape[1]
        # else:
        #     self.p_val = self.sign_level
