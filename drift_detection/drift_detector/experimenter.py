from .detector import Detector
from .synthetic_applicator import SyntheticShiftApplicator
from .clinical_applicator import ClinicalShiftApplicator
from .utils import get_args
from typing import Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
        syntheticshiftapplicator: SyntheticShiftApplicator = None,
        clinicalshiftapplicator: ClinicalShiftApplicator = None,
        rollingwindow = None,
        random_runs = 5,
        admin_data: pd.DataFrame = None,
        data: pd.DataFrame = None
    ):

        self.detector = detector
        self.syntheticshiftapplicator = syntheticshiftapplicator
        self.clinicalshiftapplicator = clinicalshiftapplicator
        self.random_runs = random_runs
        self.admin_data = admin_data
        
        self.samples = [10, 20, 50, 100, 200, 500, 1000]
        
    def apply_synthetic_shift(
        self,
        X: np.ndarray, 
        **kwargs
    ):         
        X_target = None
        
        if self.syntheticshiftapplicator is not None:
            X_target, _ = self.syntheticshiftapplicator.apply_shift(
                X, 
                **kwargs
            )
        else:
            raise ValueError("No SyntheticShiftApplicator detected.")
        return X_target    
        
    def apply_clinical_shift(
        self, 
        X: pd.DataFrame,
        **kwargs
    ):
        X_source = None
        X_target = None
        
        if self.clinicalshiftapplicator is not None:
            X_source, X_target = self.clinicalshiftapplicator.apply_shift(
                X, 
                self.admin_data, 
                **kwargs
            )
        else:
            raise ValueError("No ClinicalShiftApplicator detected.")
        return X_source, X_target
        
    def detect_shift_sample(
        self, 
        X_target: Union[np.ndarray, torch.utils.data.Dataset], 
        sample: int, 
        **kwargs
    ):
        """
        Tests shift between source and target data.
        Parameters
        ----------
        X_source : np.ndarray
            Source data.
        X_target : np.ndarray
            Target data.
        **kwargs
            Keyword arguments for Detector.
        Returns
        -------
        dict
            Dictionary containing p-value and distance.
        """
        
        drift_results = self.detector.detect_shift(
            X_target, 
            sample,
            **get_args(self.detector.detect_shift, kwargs)
        )

        return drift_results

    def detect_shift_samples(
        self,
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
        
        drift_samples_results = self.detector.detect_shift_samples(
            X_target,
            **get_args(self.detector.detect_shift, kwargs)
        )

        return drift_samples_results

