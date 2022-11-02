# from drift_detector.detector import Detector
# from drift_detector.synthetic_applicator import SyntheticShiftApplicator
# from drift_detector.clinical_applicator import ClinicalShiftApplicator
from typing import Union

import numpy as np
import pandas as pd
import torch

from drift_detection.drift_detector import (
    ClinicalShiftApplicator,
    Detector,
    SyntheticShiftApplicator,
)


class Experimenter:

    """
    Experimenter class for testing out various distribution shifts.
    Attributes
    ----------
    detector : Detector
        Detector object for detecting data shift.
    syntheticshiftapplicator : SyntheticShiftApplicator
        SyntheticShiftApplicator object for applying
        synthetic data shift to target data.
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
        shiftapplicator: Union[
            SyntheticShiftApplicator, ClinicalShiftApplicator
        ] = None,
    ):

        self.detector = detector
        self.shiftapplicator = shiftapplicator

        self.experiment_type = experiment_type

        self.experiment_types = {
            "sensitivity_test": self.sensitivity_test,
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

    def run(self, X: Union[np.ndarray, torch.utils.data.Dataset]):
        if isinstance(X, torch.utils.data.Dataset):
            X, _ = self.detector.transform(X)
        if self.shiftapplicator is not None:
            X_target, _ = self.shiftapplicator.apply_shift(X)
        else:
            X_target = X

        drift_sample_results = self.experiment_types[self.experiment_type](X_target)

        return drift_sample_results

    def apply_clinical_shift(self, X: pd.DataFrame, **kwargs):
        X_source = None
        X_target = None

        if self.clinicalshiftapplicator is not None:
            X_source, X_target = self.clinicalshiftapplicator.apply_shift(
                X, self.admin_data, **kwargs
            )
        else:
            raise ValueError("No ClinicalShiftApplicator detected.")
        return X_source, X_target

    def sensitivity_test(self, X_target):
        drift_samples_results = self.detector.detect_shift_samples(X_target)
        return drift_samples_results

    def balanced_sensitivity_test(
        self,
    ):
        raise NotImplementedError

    def rolling_window_drift(
        self,
    ):
        raise NotImplementedError

    def rolling_window_performance(
        self,
    ):
        raise NotImplementedError
