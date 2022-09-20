from drift_detection.drift_detector import Reductor, TSTester, DCTester
from typing import List, Tuple, Union, Optional, Callable, Any, Dict
from drift_detection.utils.drift_detector_utils import get_args
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import torch

@hydra.main(config_path="../configs/detector", config_name="NIHCXR-txrv_ae-mmd")
class Detector:

    """
    Detector class for distribution shift detection. 


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
    detect_shift(source_data, target_data, **kwargs)
        Detects shift between source and target data.
    """
    def __init__(
        self,
        reductor: Reductor = None,
        tester: Union[TSTester, DCTester] = None,
        p_val_threshold: float = 0.05
    ):
        print(OmegaConf.to_yaml(cfg.detector))

        self.reductor = reductor
        self.tester = tester
        self.p_val_threshold = p_val_threshold

    def fit(self, data: Union[np.ndarray, torch.utils.data.Dataset]):
        self.reductor.fit(data)

    def transform(self, X, **kwargs):
        return self.reductor.transform(X, **kwargs)
        
    def test_shift(self, X_s, X_t, **kwargs):
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