"""Detector base class."""

from typing import Union, List

import numpy as np
import torch
from tqdm import tqdm

from cyclops.monitor.reductor import Reductor
from cyclops.monitor.tester import DCTester, TSTester

from cyclops.monitor.utils import get_args, get_device

from datasets.arrow_dataset import Dataset


class Detector:
    """Detector class for distribution shift detection.

    Detector combines a Reductor and a Tester to detect distribution shift.

    Attributes
    ----------
    reductor : Reductor
        Reductor object for dimensionality reduction.
    tester : TSTester or DCTester
        Tester object for statistical testing.
    p_val_threshold : float
        Threshold for p-value. If p-value is below this threshold, a shift is detected.

    """

    def __init__(
        self,
        experiment_type: str,
        reductor: Reductor,
        tester: Union[TSTester, DCTester],
        device: str =None,
        **kwargs
    ):

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
        
        self.reductor = reductor
        self.tester = tester
        if device is None:
            self.device = get_device()
        else:
            self.device = device


    def fit(self, ds_source: Dataset, **kwargs):
        """Fit Reductor to data."""
        self.reductor.fit(ds_source)

        source_features = self.transform(
            ds_source, **get_args(self.reductor.transform, kwargs)
        )

        self.tester.fit(source_features, **get_args(self.tester.fit, kwargs))

    def transform(self, dataset: Dataset, **kwargs):
        """Transform data.

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
        return self.reductor.transform(dataset, device=self.device, **kwargs)

    def test_shift(self, X_target):
        """Test shift between source and target data.

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
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        p_val, dist = self.tester.test_shift(X_target)

        if p_val < self.p_val_threshold:
            shift_detected = 1
        else:
            shift_detected = 0

        return {"p_val": p_val, "distance": dist, "shift_detected": shift_detected}

    def detect_shift(
        self,
        ds_source: Dataset,
        ds_target: Dataset
    ):
        """Detect shift between source and target data.

        Parameters
        ----------
        ds_target: Dataset
            Target dataset.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        drift_sample_results = self.experiment_types[self.experiment_type](ds_source, ds_target)


    # def detect_shift_samples(
    #     self, ds_target: Dataset, **kwargs
    # ):
    #     """Detect shift between source and target data across samples.

    #     Parameters
    #     ----------
    #     X_source: np.ndarray or torch.utils.data.Dataset
    #         Source data.
    #     X_target: np.ndarray or torch.utils.data.Dataset
    #         Target data.
    #     **kwargs
    #         Keyword arguments for Reductor and TSTester.

    #     Returns
    #     -------
    #     dict
    #         Dictionary containing p-value, distance, and boolean 'shift_detected'.

    #     """
    #     p_val_samples = np.ones((len(self.samples), self.random_runs)) * (-1)
    #     dist_samples = np.ones((len(self.samples), self.random_runs)) * (-1)

    #     pbar_total = self.random_runs * len(self.samples)
    #     with tqdm(total=pbar_total, miniters=int(pbar_total / 100)) as pbar:
    #         for rand_run in range(self.random_runs):
    #             # np.random.seed(rand_run)
    #             # np.random.shuffle(X_target)

    #             for sample_iter, sample in enumerate(self.samples):
    #                 drift_results = self.detect_shift(ds_target, sample, **kwargs)

    #                 p_val_samples[sample_iter, rand_run] = drift_results["p_val"]
    #                 dist_samples[sample_iter, rand_run] = drift_results["distance"]

    #                 pbar.update(1)

    #     mean_p_vals = np.mean(p_val_samples, axis=1)
    #     std_p_vals = np.std(p_val_samples, axis=1)

    #     mean_dist = np.mean(dist_samples, axis=1)
    #     std_dist = np.std(dist_samples, axis=1)

    #     drift_samples_results = {
    #         "samples": self.samples,
    #         "mean_p_vals": mean_p_vals,
    #         "std_p_vals": std_p_vals,
    #         "mean_dist": mean_dist,
    #         "std_dist": std_dist,
    #     }
    #     return drift_samples_results

    def sensitivity_test(self, 
                         ds_target: Dataset, 
                         sample_size: int, **kwargs):
        """Sensitivity test for drift detection."""
        ds_target_sample = ds_target.select(np.random.choice(ds_target.shape[0], sample_size, replace=False))
        
        # get target features
        target_features = self.transform(
            ds_target_sample, **get_args(self.reductor.transform, kwargs)
        )
        results = self.test_shift(target_features, **get_args(self.tester.test_shift, kwargs))

        if results["p_val"] < self.tester.p_val_threshold:
            shift_detected = 1
        else:
            shift_detected = 0

        return {
            "p_val": results["p_val"],
            "distance": results["distance"],
            "shift_detected": shift_detected,
        }
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