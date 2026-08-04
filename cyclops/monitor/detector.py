"""Detector base class."""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from datasets import concatenate_datasets
from datasets.arrow_dataset import Dataset

from cyclops.data.slicer import SliceSpec
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
    ) -> None:
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
                Must be one of {self.experiment_types.keys()}",
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
            ds_source,
            **get_args(self.reductor.transform, kwargs),
        )

        if self.tester.tester_method == "ctx_mmd":
            kwargs["ds_source"] = ds_source
        self.tester.fit(source_features, **get_args(self.tester.fit, kwargs))

    def transform(self, dataset: Dataset) -> np.ndarray[float, np.dtype[np.float64]]:
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
        return self.reductor.transform(dataset)

    def test_shift(
        self,
        X_target: np.ndarray[float, np.dtype[np.float64]],
        **kwargs: Any,
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

        shift_detected = 1 if p_val < self.tester.p_val_threshold else 0

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
        return self.experiment_types[self.experiment_type](
            ds_source,
            ds_target,
            **self.method_args,
        )

    def _detect_shift_sample(self, ds_target: Dataset) -> Dict[str, Any]:
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
        target_features = self.transform(ds_target)
        if self.tester.tester_method == "ctx_mmd":
            results = self.test_shift(target_features, ds_target=ds_target)
        else:
            results = self.test_shift(target_features)

        shift_detected = 1 if results["p_val"] < self.tester.p_val_threshold else 0

        return {
            "p_val": results["p_val"],
            "distance": results["distance"],
            "shift_detected": shift_detected,
        }

    def detect_shift_by_subgroup(
        self,
        ds_target: Dataset,
        slice_spec: SliceSpec,
        correction: str = "bonferroni",
        min_sample_size: int = 30,
        batched: bool = True,
        batch_size: int = 1000,
        num_proc: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """Detect distribution shift independently within each subgroup.

        A model can look stable when tested against the whole target
        population while drifting badly for a specific clinically or
        socially relevant subgroup (e.g. an age band, sex, or hospital
        site) - an aggregate test can mask this. This method runs the
        already-fit tester separately on each subgroup of `ds_target`
        defined by `slice_spec`, so that subgroup-level shift can be
        detected and reported on its own, which is useful for
        health-equity-aware monitoring of deployed models.

        Parameters
        ----------
        ds_target : Dataset
            Target dataset to test for shift, split into subgroups.
        slice_spec : SliceSpec
            Specification of the subgroups (slices) of `ds_target` to test
            independently. See :class:`cyclops.data.slicer.SliceSpec`.
        correction : str, optional
            Multiple-testing correction applied to the p-value threshold
            across all subgroups tested, to control the false-positive
            rate that testing many subgroups simultaneously would
            otherwise inflate. One of "bonferroni" or "none". Default is
            "bonferroni".
        min_sample_size : int, optional
            Minimum number of samples required in a subgroup for the
            shift test to be run. Subgroups with fewer samples than this
            are still returned (with their sample size), but with
            `p_val`/`distance`/`shift_detected` set to None, since a
            statistical test on too few samples is unreliable. Default
            is 30.
        batched : bool, optional
            Whether to filter the dataset in batches. Default is True.
        batch_size : int, optional
            Batch size to use when filtering. Default is 1000.
        num_proc : int, optional
            Number of processes to use when filtering. Default is 1.

        Returns
        -------
        dict
            Dictionary mapping each subgroup's slice name to a dictionary
            with keys `p_val`, `distance`, `shift_detected`, and
            `sample_size`.

        Examples
        --------
        >>> import numpy as np
        >>> from datasets import Dataset
        >>> from cyclops.data.slicer import SliceSpec
        >>> from cyclops.monitor.detector import Detector
        >>> from cyclops.monitor.reductor import Reductor
        >>> from cyclops.monitor.tester import TSTester
        >>> np.random.seed(0)
        >>> ds_source = Dataset.from_dict(
        ...     {
        ...         "feature_0": np.random.rand(100),
        ...         "sex": ["M", "F"] * 50,
        ...     },
        ... )
        >>> ds_target = Dataset.from_dict(
        ...     {
        ...         "feature_0": np.random.rand(100),
        ...         "sex": ["M", "F"] * 50,
        ...     },
        ... )
        >>> reductor = Reductor("nored", feature_columns=["feature_0"])
        >>> tester = TSTester("mmd")
        >>> detector = Detector("sensitivity_test", reductor, tester)
        >>> detector.fit(ds_source)
        >>> slice_spec = SliceSpec(
        ...     spec_list=[{"sex": {"value": "M"}}, {"sex": {"value": "F"}}],
        ... )
        >>> results = detector.detect_shift_by_subgroup(ds_target, slice_spec)

        """
        if correction not in ("bonferroni", "none"):
            raise ValueError(
                f"Unknown correction method: {correction}. "
                "Must be one of 'bonferroni', 'none'.",
            )

        slices = slice_spec.get_slices()
        base_threshold = self.tester.p_val_threshold
        threshold = (
            base_threshold / len(slices)
            if correction == "bonferroni"
            else base_threshold
        )

        results: Dict[str, Dict[str, Any]] = {}
        for slice_name, slice_fn in slices.items():
            ds_subgroup = ds_target.filter(
                slice_fn,
                batched=batched,
                batch_size=batch_size,
                num_proc=num_proc,
            )
            sample_size = ds_subgroup.shape[0]
            if sample_size < min_sample_size:
                results[slice_name] = {
                    "p_val": None,
                    "distance": None,
                    "shift_detected": None,
                    "sample_size": sample_size,
                }
                continue

            drift_result = self._detect_shift_sample(ds_subgroup)
            results[slice_name] = {
                "p_val": drift_result["p_val"],
                "distance": drift_result["distance"],
                "shift_detected": 1 if drift_result["p_val"] < threshold else 0,
                "sample_size": sample_size,
            }
        return results

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
                np.random.choice(ds_source.shape[0], source_sample_size, replace=False),
            )
            self.fit(ds_source_sample)
            for i, sample in enumerate(target_sample_size):
                ds_target_sample = ds_target.select(
                    np.random.choice(ds_target.shape[0], sample, replace=False),
                )

                drift_results = self._detect_shift_sample(ds_target_sample)

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
                np.random.choice(ds_source.shape[0], source_sample_size, replace=False),
            )
            self.fit(ds_source_sample)
            for i, sample in enumerate(target_sample_size):
                ds_target_sample1 = ds_source.select(
                    np.random.choice(
                        ds_source.shape[0],
                        source_sample_size - sample,
                        replace=False,
                    ),
                )
                ds_target_sample2 = ds_target.select(
                    np.random.choice(ds_target.shape[0], sample, replace=False),
                )
                ds_target_balanced = concatenate_datasets(
                    [ds_target_sample1, ds_target_sample2],
                )

                drift_results = self._detect_shift_sample(ds_target_balanced)

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
            index=pd.to_datetime(ds_target[timestamp_column]),
        ).resample(window_size)
        timestamps = resampler.mean().index
        indices = list(resampler.indices.values())

        p_val = np.empty((num_runs, len(indices)))
        dist = np.empty((num_runs, len(indices)))
        shift_detected = np.empty((num_runs, len(indices)))

        for run in range(num_runs):
            ds_source_sample = ds_source.select(
                np.random.choice(ds_source.shape[0], source_sample_size, replace=False),
            )
            self.fit(ds_source_sample)
            for i, sample in enumerate(indices):
                ds_target_timestep = ds_target.select(sample)
                ds_target_sample = ds_target_timestep.select(
                    np.random.choice(
                        ds_target_timestep.shape[0],
                        target_sample_size,
                        replace=False,
                    ),
                )
                drift_results = self._detect_shift_sample(ds_target_sample)

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
