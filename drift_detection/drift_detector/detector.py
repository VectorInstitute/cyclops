from drift_detection.drift_detector import Reductor, TSTester, DCTester
from typing import List, Tuple, Union, Optional, Callable, Any, Dict
from .utils import get_args
import numpy as np
import torch
from tqdm import tqdm

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
        random_runs = 5
    ):

        self.reductor = reductor
        self.tester = tester
        self.p_val_threshold = p_val_threshold
        self.random_runs = random_runs
        self.samples = [10, 20, 50, 100, 200, 500, 1000]

    def fit(self, X_source: Union[np.ndarray, torch.utils.data.Dataset], **kwargs):
        
        
        self.reductor.fit(X_source)
        
        X_transformed = self.transform(X_source, **get_args(self.reductor.transform, kwargs))

        if isinstance(X_transformed, tuple):
            X_transformed = X_transformed[0]
        
        self.tester.fit(X_transformed, **kwargs)

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

    def test_shift(self, X_target, **kwargs):
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
        p_val, dist = self.tester.test_shift(X_target, **kwargs)

        if p_val < self.p_val_threshold:
            shift_detected = 1
        else:
            shift_detected = 0
        
        return {
            "p_val": p_val, 
            "distance": dist, 
            "shift_detected": shift_detected
        }

    def detect_shift(
        self,
        X_target: Union[np.ndarray, torch.utils.data.Dataset],
        sample: int, 
        **kwargs
    ):
        """
        Detects shift between source and target data.


        Parameters
        ----------
        X_source: np.ndarray or torch.utils.data.Dataset
            Source data.
        X_target: np.ndarray or torch.utils.data.Dataset
            Target data.
        sample: int 
            Number of sample in test set.
        **kwargs
            Keyword arguments for Reductor and TSTester.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.
        """
        
        # check if reductor_method contains 'txrv'
        if 'txrv' not in self.reductor.dr_method:
            X_t = self.transform(X_target, **get_args(self.reductor.transform, kwargs))
        else:
            X_t = X_target

        results = self.test_shift(
            X_t[:sample,:], 
            **kwargs
        )

        if results["p_val"] < self.p_val_threshold:
            shift_detected = True
        else:
            shift_detected = False

        return {
            "p_val": results["p_val"],
            "distance": results["distance"],
            "shift_detected": shift_detected,
        }
    
    def detect_shift_samples(
        self,
        X_target: Union[np.ndarray, torch.utils.data.Dataset],
        **kwargs
    ):
        """
        Detects shift between source and target data.
        Parameters
        ----------
        X_source: np.ndarray or torch.utils.data.Dataset
            Source data.
        X_target: np.ndarray or torch.utils.data.Dataset
            Target data.
        **kwargs
            Keyword arguments for Reductor and TSTester.
        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.
        """
        
        p_val_samples = np.ones((len(self.samples), self.random_runs)) * (-1)
        dist_samples = np.ones((len(self.samples), self.random_runs)) * (-1)
        
        pbar_total = self.random_runs * len(self.samples)
        with tqdm(total= pbar_total, miniters = int(pbar_total/100)) as pbar:
            for rand_run in range(self.random_runs):

                np.random.seed(rand_run)
                np.random.shuffle(X_target)

                for si, sample in enumerate(self.samples):

                    drift_results = self.detect_shift(
                        X_target,
                        sample,
                        **kwargs
                    )

                    p_val_samples[si, rand_run] = drift_results['p_val']
                    dist_samples[si, rand_run] = drift_results['distance']
                    
                    pbar.update(1)
            
        mean_p_vals = np.mean(p_val_samples, axis=1)
        std_p_vals = np.std(p_val_samples, axis=1)
    
        mean_dist = np.mean(dist_samples, axis=1)
        std_dist = np.std(dist_samples, axis=1)
        
        drift_samples_results = {
            'samples': self.samples,
            'mean_p_vals': mean_p_vals,
            'std_p_vals': std_p_vals,
            'mean_dist': mean_dist,
            'std_dist': std_dist
        }
        
        return drift_samples_results
