"""Tester Module for drift detection with TSTester and DCTester submodules."""

from typing import Any, List, Tuple

import numpy as np
import sklearn
import torch
from alibi_detect.cd import (
    ChiSquareDrift,
    ClassifierDrift,
    FETDrift,
    KSDrift,
    LSDDDrift,
    MMDDrift,
    SpotTheDiffDrift,
    TabularDrift,
)

from cyclops.monitor.utils import ContextMMDWrapper, LKWrapper, get_args


class TSTester:
    """Two Sample Statistical Tester.

    Parameters
    ----------
    tester_method: str
        two-sample statistical test method
        available methods are:
            "lk"

    Methods
    -------
    get_available_test_methods()
        Get available test methods
    fit(X_s: np.ndarray, **kwargs)
        Fit statistical test method to reference data
    test_shift(X_t: np.ndarray, **kwargs)
        Test for shift in data

    """

    def __init__(
        self,
        tester_method: str,
        p_val_threshold: float = 0.05,
        **kwargs: Any,
    ):
        self.tester_method = tester_method
        self.method: Any = None
        self.p_val_threshold = p_val_threshold

        # dict where the key is the string of each test_method
        # and the value is the class of the test_method
        self.tester_methods = {
            "ks": KSDrift,
            "chi2": ChiSquareDrift,
            "mmd": MMDDrift,
            "ctx_mmd": ContextMMDWrapper,
            "lsdd": LSDDDrift,
            "lk": LKWrapper,
            "fet": FETDrift,
            "tabular": TabularDrift,
        }

        self.method_args = kwargs
        if "backend" not in self.method_args:
            self.method_args["backend"] = "pytorch"

        if self.tester_method not in self.tester_methods:
            raise ValueError(
                f"Tester method {self.tester_method} not supported. \
                    Must be one of {self.tester_methods.keys()}"
            )

    def get_available_test_methods(self) -> List[str]:
        """Return list of available test methods."""
        return list(self.tester_methods.keys())

    def fit(self, X_s: np.ndarray[float, np.dtype[np.float64]], **kwargs: Any) -> None:
        """Initialize test method to source data."""
        X_s = X_s.astype("float32")
        # append alternative="two-sided" to method_args"
        # if not already present
        # this is required for the FET test
        # to work properly
        if self.tester_method == "fet":
            if "alternative" not in self.method_args:
                self.method_args["alternative"] = "two-sided"

        if self.tester_method == "ctx_mmd":
            if "ds_source" in kwargs:
                self.method = self.tester_methods[self.tester_method](
                    X_s,
                    ds_source=kwargs["ds_source"],
                    **get_args(
                        self.tester_methods[self.tester_method], self.method_args
                    ),
                )
            else:
                raise ValueError(
                    "ds_source must be provided to fit method \
                    for ctx_mmd."
                )
        else:
            self.method = self.tester_methods[self.tester_method](
                X_s,
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )

    def test_shift(
        self, X_t: np.ndarray[float, np.dtype[np.float64]], **kwargs: Any
    ) -> Tuple[float, float]:
        """Test for shift in data."""
        X_t = X_t.astype("float32")
        num_features = X_t.shape[1]

        if self.tester_method == "ctx_mmd":
            if "ds_target" in kwargs:
                preds = self.method.predict(
                    X_t,
                    ds_target=kwargs["ds_target"],
                    **get_args(self.method.predict, self.method_args),
                )

            else:
                raise ValueError(
                    "ds_target must be provided to test_shift method \
                    for ctx_mmd."
                )
        else:
            preds = self.method.predict(
                X_t, **get_args(self.method.predict, self.method_args)
            )

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]

        if isinstance(p_val, np.ndarray):
            idx = np.argmin(p_val)
            p_val = p_val[idx]
            dist = dist[idx]

        if self.tester_method in ["ks", "chi2", "fet", "tabular"]:
            self.p_val_threshold = self.p_val_threshold / num_features

        return p_val, dist


class DCTester:
    """Domain Classifier Tester.

    Parameters
    ----------
    tester_method: str
        domain classifier test method
        Must be one of: "spot_the_diff" or "classifier"
    p_val_threshold: float
        p-value threshold for statistical test
    model: str
        model to use for chosen test method.
        Must be either a sklearn or pytorch model.

    Methods
    -------
    get_available_test_methods()
        Get available test methods
    fit(X_s: np.ndarray, **kwargs)
        Fit domain classifier to reference data
    test_shift(X_t: np.ndarray, **kwargs)
        Test for shift in data

    """

    def __init__(
        self, tester_method: str, p_val_threshold: float = 0.05, **kwargs: Any
    ):
        self.tester_method = tester_method
        self.p_val_threshold = p_val_threshold
        self.method_args = kwargs
        self.tester: Any = None

        self.tester_methods = {
            "spot_the_diff": SpotTheDiffDrift,
            "classifier": ClassifierDrift,
        }
        if self.tester_method not in self.tester_methods:
            raise ValueError(
                f"Tester method {self.tester_method} not supported. \
                Must be one of {self.tester_methods.keys()}"
            )

    def get_available_test_methods(self) -> List[str]:
        """Return list of available test methods."""
        return list(self.tester_methods.keys())

    def fit(self, X_s: np.ndarray[float, np.dtype[np.float64]]) -> None:
        """Initialize test method to source data."""
        X_s = X_s.astype("float32")

        if self.tester_method == "spot_the_diff":
            self.tester = self.tester_methods[self.tester_method](
                X_s,
                backend="pytorch",
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )
        elif self.tester_method == "classifier":
            if isinstance(self.method_args["model"], torch.nn.Module):
                self.method_args["backend"] = "pytorch"
            elif isinstance(self.method_args["model"], sklearn.base.ClassifierMixin):
                self.method_args["backend"] = "sklearn"
            else:
                raise ValueError(
                    "Model must be one of: torch.nn.Module or \
                    sklearn.base.BaseEstimator"
                )
            self.tester = self.tester_methods[self.tester_method](
                X_s,
                **get_args(self.tester_methods[self.tester_method], self.method_args),
            )

    def test_shift(
        self, X_t: np.ndarray[float, np.dtype[np.float64]]
    ) -> Tuple[float, float]:
        """Test for shift in data."""
        X_t = X_t.astype("float32")
        preds = self.tester.predict(X_t)

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist
