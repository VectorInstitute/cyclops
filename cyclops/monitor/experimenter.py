"""Experimenter class for drift detection."""
# from drift_detector.detector import Detector
# from drift_detector.synthetic_applicator import SyntheticShiftApplicator
# from drift_detector.clinical_applicator import ClinicalShiftApplicator
from typing import Union

import numpy as np
import pandas as pd
import torch

from cyclops.monitor.clinical_applicator import ClinicalShiftApplicator
from cyclops.monitor.detector import Detector
from cyclops.monitor.synthetic_applicator import SyntheticShiftApplicator


class Experimenter:
    """Experimenter class for testing out various distribution shifts.

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

        self.experiment_types: dict = {
            "sensitivity_test": self.sensitivity_test,
            "balanced_sensitivity_test": self.balanced_sensitivity_test,
            "rolling_window_drift": self.rolling_window_drift,
            "rolling_window_performance": self.rolling_window_performance,
        }

        if self.experiment_type not in self.experiment_types:
            raise ValueError(
                f"Experiment type {self.experiment_type} not supported. \
                Must be one of {self.experiment_types.keys()}"
            )

    def run(
        self,
        X: Union[np.ndarray, torch.utils.data.Dataset],
        metadata: pd.DataFrame,
        metadata_mapping: dict,
    ):
        """Run experiment.

        Parameters
        ----------
        X : Union[np.ndarray, torch.utils.data.Dataset]
            Target data to run experiment on.

        Returns
        -------
        results : dict
            Dictionary of experiment results.

        """
        if self.shiftapplicator is not None:
            if isinstance(self.shiftapplicator, ClinicalShiftApplicator):
                X_source, X_target = self.shiftapplicator.apply_shift(
                    X, metadata, metadata_mapping
                )
                self.detector.fit(X_source, progress=False)
                X_target, _ = self.detector.transform(X_target)
            else:
                self.detector.fit(X, progress=False)
                if isinstance(X, torch.utils.data.Dataset):
                    X, _ = self.detector.transform(X)
                X_target, _ = self.shiftapplicator.apply_shift(
                    X, metadata, metadata_mapping
                )
        else:
            self.detector.fit(X, progress=False)
            if isinstance(X, torch.utils.data.Dataset):
                X, _ = self.detector.transform(X)
            X_target = X

        drift_sample_results = self.experiment_types[self.experiment_type](X_target)

        return drift_sample_results

    def sensitivity_test(self, X_target):
        """Sensitivity test for drift detection."""
        drift_samples_results = self.detector.detect_shift_samples(X_target)
        return drift_samples_results

    def balanced_sensitivity_test(
        self,
    ):
        """Perform balanced sensitivity test for drift detection."""
        raise NotImplementedError

    def rolling_window_drift(
        self,
    ):
        """Perform rolling window drift test for drift detection."""
        raise NotImplementedError

    def rolling_window_performance(
        self,
    ):
        """Perform rolling window performance test for drift detection."""
        raise NotImplementedError
