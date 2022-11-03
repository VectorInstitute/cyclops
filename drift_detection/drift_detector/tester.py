"""Tester Module for drift detection with TSTester and DCTester submodules."""
from alibi_detect.cd import (
    ChiSquareDrift,
    ClassifierDrift,
    ClassifierUncertaintyDrift,
    FETDrift,
    KSDrift,
    LSDDDrift,
    MMDDrift,
    SpotTheDiffDrift,
    TabularDrift,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from drift_detection.drift_detector.utils import (
    ContextMMDWrapper,
    LKWrapper,
    convolutional_neural_network,
    feed_forward_neural_network,
    get_args,
    recurrent_neural_network,
)


class TSTester:
    """Two Sample Statistical Tester.

    Parameters
    ----------
    tester_method: str
        two-sample statistical test method

    Methods
    -------
    get_available_test_methods()
        Get available test methods
    fit(X_s: np.ndarray, **kwargs)
        Fit statistical test method to reference data
    test_shift(X_t: np.ndarray, **kwargs)
        Test for shift in data

    """

    def __init__(self, tester_method: str):
        self.tester_method = tester_method
        self.method = None

        # dict where the key is the string of each test_method
        # and the value is the class of the test_method
        self.tester_methods = {
            "lk": LKWrapper,
            "lsdd": LSDDDrift,
            "mmd": MMDDrift,
            "tabular": TabularDrift,
            "ctx_mmd": ContextMMDWrapper,
            "chi2": ChiSquareDrift,
            "fet": FETDrift,
            "ks": KSDrift,
        }

        if self.tester_method not in self.tester_methods:
            raise ValueError(
                f"Tester method {self.tester_method} not supported. \
                    Must be one of {self.tester_methods.keys()}"
            )

    def get_available_test_methods(self):
        """Return list of available test methods."""
        return list(self.tester_methods.keys())

    def fit(self, X_s, **kwargs):
        """Initialize test method to source data."""
        X_s = X_s.astype("float32")

        self.method = self.tester_methods[self.tester_method](
            X_s, **get_args(self.tester_methods[self.tester_method], kwargs)
        )

    def test_shift(self, X_t, **kwargs):
        """Test for shift in data."""
        X_t = X_t.astype("float32")

        preds = self.method.predict(X_t, **get_args(self.method.predict, kwargs))

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist


class DCTester:
    """Domain Classifier Tester.

    Parameters
    ----------
    model: str
        model to use for domain classification.
        Must be one of: "gb", "rf", "rnn", "cnn", "ffnn"

    Methods
    -------
    get_available_model_methods()
        Get available model methods
    fit(X_s: np.ndarray, **kwargs)
        Fit domain classifier to reference data
    test_shift(X_t: np.ndarray, **kwargs)
        Test for shift in data

    """

    def __init__(self, tester_method: str, model_method: str = None):
        self.tester_method = tester_method
        self.model_method = model_method
        self.tester = None
        self.model = None

        self.tester_methods = {
            "spot_the_diff": SpotTheDiffDrift,
            "classifier": ClassifierDrift,
            "classifier_uncertainty": ClassifierUncertaintyDrift,
        }
        self.model_methods = {
            "gb": GradientBoostingClassifier,
            "rf": RandomForestClassifier,
            "rnn": recurrent_neural_network,
            "cnn": convolutional_neural_network,
            "ffnn": feed_forward_neural_network,
        }

        if self.tester_method not in self.tester_methods:
            raise ValueError(
                f"Tester method {self.tester_method} not supported. \
                Must be one of {self.tester_methods.keys()}"
            )

        if self.model_method not in self.model_methods:
            raise ValueError(
                f"Model method {self.model_method} not supported.\
                Must be one of {self.model_methods.keys()}"
            )

    def get_available_test_methods(self):
        """Return list of available test methods."""
        return list(self.tester_methods.keys())

    def get_available_model_methods(self):
        """Return list of available model methods."""
        return list(self.model_methods.keys())

    def fit(self, X_s, **kwargs):
        """Initialize test method to source data."""
        X_s = X_s.astype("float32")

        if self.tester_method == "spot_the_diff":
            self.tester = self.tester_methods[self.tester_method](
                X_s, **get_args(self.tester_methods[self.tester_method], kwargs)
            )
        else:
            self.model = self.model_methods[self.model_method](
                **get_args(self.model_methods[self.model_method], kwargs)
            )
            self.tester = self.tester_methods[self.tester_method](
                X_s,
                self.model,
                **get_args(self.tester_methods[self.tester_method], kwargs),
            )

    def test_shift(self, X_t):
        """Test for shift in data."""
        X_t = X_t.astype("float32")

        preds = self.tester.predict(X_t)

        p_val = preds["data"]["p_val"]
        dist = preds["data"]["distance"]
        return p_val, dist
