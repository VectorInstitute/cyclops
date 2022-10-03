from drift_detection.drift_detector import Detector, SyntheticShiftApplicator, ClinicalShiftApplicator
from typing import List, Tuple, Union, Optional, Callable, Any, Dict
from drift_detection.drift_detector.utils import get_args
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
        shiftapplicator: Union[SyntheticShiftApplicator, ClinicalShiftApplicator] = None
    ):

        self.detector = detector
        self.shiftapplicator = shiftapplicator
        
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
            Keyword arguments for Tester.
        Returns
        -------
        dict
            Dictionary containing p-value and distance.
        """
        p_val, dist = self.tester.test_shift(X_s, X_t, **kwargs)

        return {"p_val": p_val, "distance": dist}

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

        p_val_samples = {}
        dist_samples = {}
        
        for sample in self.samples:
            
            if isinstance(self.shiftapplicator, SyntheticShiftApplicator):
                X_shifted_target, _ = self.shiftapplicator.apply_shift(target_data)
            elif isinstance(self.shiftapplicator, ClinicalShiftApplicator):
                X_source, y_source, X_shifted_target, y_shifted_target = self.shiftapplicator.apply_shift()  
            else: 
                raise ValueError("No Shift Applicator provided, set Shift Applicator to apply shift.")
            
            p_val, std = self.detector.detect_shift(X_source[:1000,:], X_shifted_target[:sample,:] **get_args(self.detector.detect_shift, kwargs))
            
            p_val_samples.update({sample: p_val})
            dist_samples.update({sample: std})

        drift_samples = {k: {'p_val': p_val_samples[k], 'distance': dist_samples[k]} for k in p_val_samples.keys() & dist_samples.keys()}

        return drift_samples
