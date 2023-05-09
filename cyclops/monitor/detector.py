"""Detector base class."""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset

from cyclops.monitor.reductor import Reductor
from cyclops.monitor.tester import DCTester, TSTester
from cyclops.monitor.utils import get_args


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
    device : str
        Device to use for testing. If None, will use GPU if available, else CPU.
    experiment_type : str
        Experiment type to run. Must be one of:
            "sensitivity_test"
            "balanced_sensitivity_test"
            "rolling_window_drift"
            "rolling_window_performance"
    experiment_types : dict
        Dictionary of experiment types and their corresponding methods.

    Methods
    -------
    sensitivity_test
        Run sensitivity test.
    balanced_sensitivity_test
        Run balanced sensitivity test.
    rolling_window_drift
        Run rolling window drift detection.

    """

    def __init__(
        self,
        experiment_type: str,
        reductor: Reductor,
        tester: Union[TSTester, DCTester],
        **kwargs: Any,
    ):
        """Initialize Detector object."""
        self.experiment_type = experiment_type

        self.experiment_types: Dict[str, Any] = {
            "sensitivity_test": self.sensitivity_test,
            "balanced_sensitivity_test": self.balanced_sensitivity_test,
            "rolling_window_drift": self.rolling_window_drift,
        }

        if self.experiment_type not in self.experiment_types:
            raise ValueError(
                f"Experiment type {self.experiment_type} not supported. \
                Must be one of {self.experiment_types.keys()}"
            )

        self.reductor = reductor
        self.tester = tester
        self.method_args = get_args(self.experiment_types[self.experiment_type], kwargs)

    def fit(self, ds_source: Dataset, **kwargs: Any) -> None:
        """Fit Reductor and Tester to source data.

        Parameters
        ----------
        X_source : Dataset
            Source dataset.
        **kwargs
            Keyword arguments for Reductor.

        """
        self.reductor.fit(ds_source)

        source_features = self.transform(
            ds_source, **get_args(self.reductor.transform, kwargs)
        )

        if self.tester.tester_method == "ctx_mmd":
            kwargs["ds_source"] = ds_source
        self.tester.fit(source_features, **get_args(self.tester.fit, kwargs))

    def transform(
        self, dataset: Dataset, batch_size: int = 32, num_workers: int = 1
    ) -> np.ndarray[float, np.dtype[np.float64]]:
        """Transform data.

        Parameters
        ----------
        dataset: Dataset
            Dataset to transform.
        batch_size: int
            Batch size for data loader.
        num_workers: int
            Number of workers for data loader.

        Returns
        -------
        np.ndarray
            Transformed data.

        """
        return self.reductor.transform(dataset, batch_size, num_workers)

    def test_shift(
        self, X_target: np.ndarray[float, np.dtype[np.float64]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Test shift between source and target data.

        Parameters
        ----------
        X_target : np.ndarray
            Target data.
        **kwargs
            Keyword arguments for tester.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        p_val, dist = self.tester.test_shift(X_target, **kwargs)

        if p_val < self.tester.p_val_threshold:
            shift_detected = 1
        else:
            shift_detected = 0

        return {"p_val": p_val, "distance": dist, "shift_detected": shift_detected}

    def detect_shift(self, ds_source: Dataset, ds_target: Dataset) -> Any:
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
        drift_sample_results = self.experiment_types[self.experiment_type](
            ds_source, ds_target, **self.method_args
        )
        return drift_sample_results

    def _detect_shift_sample(
        self,
        ds_target: Dataset,
        batch_size: int,
        num_workers: int,
    ) -> Dict[str, Any]:
        """Detect shift between source and target data across samples.

        Parameters
        ----------
        ds_target: Dataset
            target dataset.
        **kwargs
            Keyword arguments for Reductor and TSTester.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        # get target features
        target_features = self.transform(ds_target, batch_size, num_workers)
        if self.tester.tester_method == "ctx_mmd":
            results = self.test_shift(target_features, ds_target=ds_target)
        else:
            results = self.test_shift(target_features)

        if results["p_val"] < self.tester.p_val_threshold:
            shift_detected = 1
        else:
            shift_detected = 0

        return {
            "p_val": results["p_val"],
            "distance": results["distance"],
            "shift_detected": shift_detected,
        }

    def sensitivity_test(
        self,
        ds_source: Dataset,
        ds_target: Dataset,
        source_sample_size: int,
        target_sample_size: Union[int, List[int]],
        num_runs: int = 1,
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> Dict[str, Any]:
        """Sensitivity test for drift detection.

        Parameters
        ----------
        ds_source: Dataset
            Source dataset.
        ds_target: Dataset
            Target dataset.
        source_sample_size: int
            Size of source sample.
        target_sample_size: int or list of int
            Size of target sample.
        num_runs: int
            Number of runs.
        batch_size: int
            Batch size for data loader.
        num_workers: int
            Number of workers for data loader.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        if isinstance(target_sample_size, int):
            target_sample_size = [target_sample_size]
        p_val = np.empty((num_runs, len(target_sample_size)))
        dist = np.empty((num_runs, len(target_sample_size)))
        shift_detected = np.empty((num_runs, len(target_sample_size)))

        for run in range(num_runs):
            ds_source_sample = ds_source.select(
                np.random.choice(ds_source.shape[0], source_sample_size, replace=False)
            )
            self.fit(ds_source_sample)
            for i, sample in enumerate(target_sample_size):
                ds_target_sample = ds_target.select(
                    np.random.choice(ds_target.shape[0], sample, replace=False)
                )

                drift_results = self._detect_shift_sample(
                    ds_target_sample, batch_size, num_workers
                )

                p_val[run, i] = drift_results["p_val"]
                dist[run, i] = drift_results["distance"]
                shift_detected[run, i] = drift_results["shift_detected"]

        p_val_threshold = self.tester.p_val_threshold
        return {
            "p_val": p_val,
            "distance": dist,
            "shift_detected": shift_detected,
            "samples": target_sample_size,
            "p_val_threshold": p_val_threshold,
        }

    def balanced_sensitivity_test(
        self,
        ds_source: Dataset,
        ds_target: Dataset,
        source_sample_size: int,
        target_sample_size: Union[int, List[int]],
        num_runs: int = 1,
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> Dict[str, Any]:
        """Perform balanced sensitivity test for drift detection.

        Parameters
        ----------
        ds_source: Dataset
            Source dataset.
        ds_target: Dataset
            Target dataset.
        source_sample_size: int
            Size of source sample.
        target_sample_size: int or list of int
            Size of target sample.
        num_runs: int
            Number of runs.
        batch_size: int
            Batch size for data loader.
        num_workers: int
            Number of workers for data loader.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        if isinstance(target_sample_size, int):
            target_sample_size = [target_sample_size]
        p_val = np.empty((num_runs, len(target_sample_size)))
        dist = np.empty((num_runs, len(target_sample_size)))
        shift_detected = np.empty((num_runs, len(target_sample_size)))

        for run in range(num_runs):
            ds_source_sample = ds_source.select(
                np.random.choice(ds_source.shape[0], source_sample_size, replace=False)
            )
            self.fit(ds_source_sample)
            for i, sample in enumerate(target_sample_size):
                ds_target_sample1 = ds_source.select(
                    np.random.choice(
                        ds_source.shape[0], source_sample_size - sample, replace=False
                    )
                )
                ds_target_sample2 = ds_target.select(
                    np.random.choice(ds_target.shape[0], sample, replace=False)
                )
                ds_target_balanced = concatenate_datasets(
                    ds_target_sample1, ds_target_sample2
                )

                drift_results = self._detect_shift_sample(
                    ds_target_balanced, batch_size, num_workers
                )

                p_val[run, i] = drift_results["p_val"]
                dist[run, i] = drift_results["distance"]
                shift_detected[run, i] = drift_results["shift_detected"]

        p_val_threshold = self.tester.p_val_threshold
        return {
            "p_val": p_val,
            "distance": dist,
            "shift_detected": shift_detected,
            "samples": target_sample_size,
            "p_val_threshold": p_val_threshold,
        }

    def rolling_window_drift(
        self,
        ds_source: Dataset,
        ds_target: Dataset,
        source_sample_size: int,
        target_sample_size: int,
        timestamp_column: str,
        window_size: str,
        num_runs: int = 1,
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> Dict[str, Any]:
        """Perform rolling window drift detection.

        Parameters
        ----------
        ds_source: Dataset
            Source dataset.
        ds_target: Dataset
            Target dataset.
        source_sample_size: int
            Size of source sample.
        target_sample_size: int
            Size of target sample.
        timestamp_column: str
            Name of timestamp column.
        window_size: str
            Size of window.
        num_runs: int
            Number of runs.
        batch_size: int
            Batch size for data loader.
        num_workers: int
            Number of workers for data loader.

        Returns
        -------
        dict
            Dictionary containing p-value, distance, and boolean 'shift_detected'.

        """
        resampler = pd.DataFrame(
            index=pd.to_datetime(ds_target[timestamp_column])
        ).resample(window_size)
        timestamps = resampler.mean().index
        indices = list(resampler.indices.values())

        p_val = np.empty((num_runs, len(indices)))
        dist = np.empty((num_runs, len(indices)))
        shift_detected = np.empty((num_runs, len(indices)))

        for run in range(num_runs):
            ds_source_sample = ds_source.select(
                np.random.choice(ds_source.shape[0], source_sample_size, replace=False)
            )
            self.fit(ds_source_sample)
            for i, sample in enumerate(indices):
                ds_target_timestep = ds_target.select(sample)
                ds_target_sample = ds_target_timestep.select(
                    np.random.choice(
                        ds_target_timestep.shape[0], target_sample_size, replace=False
                    )
                )
                drift_results = self._detect_shift_sample(
                    ds_target_sample, batch_size, num_workers
                )

                p_val[run, i] = drift_results["p_val"]
                dist[run, i] = drift_results["distance"]
                shift_detected[run, i] = drift_results["shift_detected"]

        p_val_threshold = self.tester.p_val_threshold
        return {
            "p_val": p_val,
            "distance": dist,
            "shift_detected": shift_detected,
            "samples": timestamps,
            "p_val_threshold": p_val_threshold,
        }
