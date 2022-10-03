from .detector import Detector
from .synthetic_applicator import SyntheticShiftApplicator
from .clinical_applicator import ClinicalShiftApplicator
from .utils import get_args
from typing import Union
import numpy as np
import torch

class Experimenter:

    """
    Experimenter class for testing out various distribution shifts.
    Attributes
    ----------
    detector : Detector
        Detector object for detecting data shift.
    syntheticshiftapplicator : SyntheticShiftApplicator
        SyntheticShiftApplicator object for applying synthetic data shift to target data.
    Methods
    -------
    detect_shift_sample(X_s, X_t, **kwargs)
        Tests shift between source and target data.
    detect_shift_samples(source_data, target_data, **kwargs)
        Detects shift between source data and target data.
    """

    def __init__(
        self,
        detector: Detector = None,
        shiftapplicator: Union[SyntheticShiftApplicator, ClinicalShiftApplicator] = None,
        random_runs = 10
    ):

        self.detector = detector
        self.shiftapplicator = shiftapplicator
        self.random_runs = random_runs
        
        self.samples = [10, 20, 50, 100, 200, 500, 1000]
        
    def detect_shift_sample(
        self, 
        X_s, 
        X_t, 
        sample: int, 
        **kwargs
    ):
        """
        Tests shift between source and target data.
        Parameters
        ----------
        X_s : np.ndarray
            Source data.
        X_t : np.ndarray
            Target data.
        **kwargs
            Keyword arguments for Detector.
        Returns
        -------
        dict
            Dictionary containing p-value and distance.
        """
        drift_results = self.detector.detect_shift(X_s, X_t, **get_args(self.detector.detect_shift, kwargs))

        return drift_results

    def detect_shift_samples(
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
        
        p_val_samples = np.ones((len(self.samples), self.random_runs)) * (-1)
        dist_samples = np.ones((len(self.samples), self.random_runs)) * (-1)
        
        for rand_run in range(0, self.random_runs):
            
            np.random.seed(rand_run)
            np.random.shuffle(X_target)
            
            ## add functionality to shuffle data
        
            for si, sample in enumerate(self.samples):
            
                if isinstance(self.shiftapplicator, SyntheticShiftApplicator):
                    X_shifted_target, _ = self.shiftapplicator.apply_shift(X_target, **kwargs)
                elif isinstance(self.shiftapplicator, ClinicalShiftApplicator):
                    X_source, y_source, X_shifted_target, y_shifted_target = self.shiftapplicator.apply_shift(X_target, **kwargs)
                else: 
                    X_shifted_target = X_target     

                drift_results = self.detector.detect_shift(X_source[:1000,:], X_shifted_target[:sample,:], **get_args(self.detector.detect_shift, kwargs))

                p_val_samples[si, rand_run] = drift_results['p_val']
                dist_samples[si, rand_run] = drift_results['distance']
            
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
