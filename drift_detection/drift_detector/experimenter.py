# from drift_detector.detector import Detector
# from drift_detector.synthetic_applicator import SyntheticShiftApplicator
# from drift_detector.clinical_applicator import ClinicalShiftApplicator
from drift_detection.drift_detector.utils import get_args
from drift_detection.drift_detector import Detector, SyntheticShiftApplicator, ClinicalShiftApplicator
from drift_detection.drift_detector.dataframe_mapping import DataFrameMapping
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
        experiment_type: str,
        detector: Detector,
        shiftapplicator: Union[SyntheticShiftApplicator, ClinicalShiftApplicator] = None,
        metadata: pd.DataFrame = None,
        metadata_mapping: DataFrameMapping = None,
        **kwargs
    ):

        
        self.detector = detector
        self.shiftapplicator = shiftapplicator
        self.metadata = metadata
        self.metadata_mapping = metadata_mapping
        
        self.random_runs = 5
        self.samples = [10, 20, 50, 100, 200, 500, 1000]

        self.experiment_type = experiment_type
        
        self.experiment_types = { 
            'sensitivity_test': self.sensitivity_test,
            "balanced_sensitivity_test": self.balanced_sensitivity_test,
            "rolling_window_drift": self.rolling_window_drift,
            "rolling_window_performance": self.rolling_window_performance,
        }
           
        if self.experiment_type not in self.experiment_types.keys():
            raise ValueError(
                "Experiment not supported, must be one of: {}".format(
                    self.experiment_types.keys()
                )
            )

    def run(self, X):
        if isinstance(X, torch.utils.data.Dataset):
            X, _ = self.detector.transform(X)
        if self.shiftapplicator is not None:
            X_target, _ = self.shiftapplicator.apply_shift(X)
        else:
            X_target = X

        drift_sample_results = self.experiment_types[self.experiment_type](X_target)

        return drift_sample_results
        
    # def apply_synthetic_shift(
    #     self,
    #     X: np.ndarray, 
    #     **kwargs
    # ):         
    #     X_target = None
        
    #     if self.syntheticshiftapplicator is not None:
    #         X_target, _ = self.syntheticshiftapplicator.apply_shift(
    #             X, 
    #             **kwargs
    #         )
    #     else:
    #         raise ValueError("No SyntheticShiftApplicator detected.")
    #     return X_target    
        
    # def apply_clinical_shift(
    #     self, 
    #     X: pd.DataFrame,
    #     **kwargs
    # ):
    #     X_source = None
    #     X_target = None
        
    #     if self.clinicalshiftapplicator is not None:
    #         X_source, X_target = self.clinicalshiftapplicator.apply_shift(
    #             X, 
    #             self.admin_data, 
    #             **kwargs
    #         )
    #     else:
    #         raise ValueError("No ClinicalShiftApplicator detected.")
    #     return X_source, X_target
        
    # def detect_shift_sample(
    #     self, 
    #     X_target: Union[np.ndarray, torch.utils.data.Dataset], 
    #     sample: int, 
    #     **kwargs
    # ):
    #     """
    #     Tests shift between source and target data.
    #     Parameters
    #     ----------
    #     X_source : np.ndarray
    #         Source data.
    #     X_target : np.ndarray
    #         Target data.
    #     **kwargs
    #         Keyword arguments for Detector.
    #     Returns
    #     -------
    #     dict
    #         Dictionary containing p-value and distance.
    #     """
        
    #     drift_results = self.detector.detect_shift(
    #         X_target, 
    #         sample,
    #         **get_args(self.detector.detect_shift, kwargs)
    #     )

    #     return drift_results

    # def detect_shift_samples(
    #     self,
    #     X_target: Union[np.ndarray, torch.utils.data.Dataset],
    #     **kwargs
    # ):
    #     """
    #     Detects shift between source and target data.
    #     Parameters
    #     ----------
    #     X_source : np.ndarray or torch.utils.data.Dataset
    #         Source data.
    #     X_target : np.ndarray or torch.utils.data.Dataset
    #         Target data.
    #     **kwargs
    #         Keyword arguments for Reductor and TSTester.
    #     Returns
    #     -------
    #     dict
    #         Dictionary containing p-value, distance, and boolean 'shift_detected'.
    #     """
        
    #     drift_samples_results = self.detector.detect_shift_samples(
    #         X_target,
    #         **get_args(self.detector.detect_shift, kwargs)
    #     )

    #     return drift_samples_results
    


    def sensitivity_test(self, X_target, **kwargs):
        drift_samples_results = self.detector.detect_shift_samples(
            X_target,
            **get_args(self.detector.detect_shift, kwargs)
        )
        return drift_samples_results

    def balanced_sensitivity_test(self,):
        pass

    def rolling_window_drift(self,):
        return
    
    def rolling_window_performance(self,):
        return
