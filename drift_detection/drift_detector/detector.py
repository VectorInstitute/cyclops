from drift_detection.drift_detector import Reductor, TSTester, DCTester
from typing import List, Tuple, Union, Optional, Callable, Any, Dict
from .utils import get_args
import numpy as np
import torch


class Detector:

    """
    ShiftDetector class for distribution shift detection.


    Attributes
    ----------
    reductor : Reductor
        Reductor object for dimensionality reduction.
    tester : TSTester or DCTester
        Tester object for statistical testing.
    p_val_threshold : float
        Threshold for p-value. If p-value is below this threshold, a shift is detected.


    Methods
    -------
    fit(X)
        Fits Reductor to data.
    transform(X, **kwargs)
        Transforms data.
    test_shift(X_source, X_target, **kwargs)
        Tests shift between source and target data.
    detect_shift(X_source, X_target, **kwargs)
        Detects shift between source data and target data.
    """

    def __init__(
        self,
        reductor: Reductor = None,
        tester: Union[TSTester, DCTester] = None,
        p_val_threshold: float = 0.05,
    ):

        self.reductor = reductor
        self.tester = tester
        self.p_val_threshold = p_val_threshold

    def fit(self, X: Union[np.ndarray, torch.utils.data.Dataset]):
        self.reductor.fit(X)

    def transform(self, X, **kwargs):
        """
        Transforms data.

        Parameters
        ----------
        X : np.ndarray or torch.utils.data.Dataset
            Data to be transformed.
        **kwargs
            Keyword arguments for Reductor.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.reductor.transform(X, **kwargs)

    def test_shift(self, X_source, X_target, **kwargs):
        """
        Tests shift between source and target data.

        Parameters
        ----------
        X_source : np.ndarray
            Source data.
        X_target : np.ndarray
            Target data.
        **kwargs
            Keyword arguments for Tester.

        Returns
        -------
        dict
            Dictionary containing p-value and distance.
        """
        p_val, dist = self.tester.test_shift(X_source, X_target, **kwargs)

        if p_val < self.p_val_threshold:
            shift_detected = True
        else:
            shift_detected = False
        
        return {
            "p_val": p_val, 
            "distance": dist, 
            "shift_detected": shift_detected
        }

    def detect_shift(
        self,
        X_source: Union[np.ndarray, torch.utils.data.Dataset],
        X_target: Union[np.ndarray, torch.utils.data.Dataset],
        **kwargs
    ):
        """
        Detects shift between source and target data.


        Parameters
        ----------
        X_source : np.ndarray or torch.utils.data.Dataset
            Source data.
        X_target : np.ndarray or torch.utils.data.Dataset
            Target data.
        **kwargs
            Keyword arguments for Reductor and TSTester.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.
        """
        
        #add if reductor is not fit then 
        #    self.fit(X_source)
        
        X_s = self.transform(X_source, **get_args(self.reductor.transform, kwargs))
        X_t = self.transform(X_target, **get_args(self.reductor.transform, kwargs))

        results = self.test_shift(X_s, X_t, **get_args(self.tester.test_shift, kwargs))

        if results["p_val"] < self.p_val_threshold:
            shift_detected = True
        else:
            shift_detected = False

        return {
            "p_val": results["p_val"],
            "distance": results["distance"],
            "shift_detected": shift_detected,
        }
