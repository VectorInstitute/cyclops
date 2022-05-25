import numpy as np
from sklearn.utils import check_array, check_random_state


class FeatureShiftDetector:
    """The super class for feature shift detection. This performs bootstrapping using the specified bootstrapping
    method {'time', 'simple'} and the specified statistic. After this, it uses the thresholds learned during
    bootstrapping to detect and localize shifted features.
    Parameters
    ----------
    statistic: divergence object, {FisherDivergence(), ModelKS(), KnnKS()}
        The divergence used during bootstrapping and detection/localization
    bootstrap_method: string, {'simple', 'time'}
        The bootstrap method used when fitting the shift detector. Simple bootstrap is the traditional bootstrapping
        where X_boot and Y_boot are randomly drawn from the concatenation of X and Y. Time bootstrap is a time
        aware variant of bootstrapping where X_boot and Y_boot are a contiguous time-series chunk which is randomly
        sampled from a clean training set.
    data_transform: pre-processing function, optional
        An optional pre-processing function called on to be called on the data before testing
    n_bootstrap_samples: int
        The number of bootstrap runs to perform when bootstrapping (i.e. {X_boot, Y_boot}
    n_window_samples: int
        The number of samples to be used when sampling X and Y. Used only for when bootstrap_method is 'time'
    n_compromised: int
        The fixed budget of features which can be checked if a shift is detected. (i.e. the number of features suspected
        to have been compromised)"""

    def __init__(
        self,
        statistic,
        bootstrap_method,
        data_transform=None,
        n_bootstrap_samples=500,
        n_window_samples=1000,
        n_compromised=1,
        significance_level=0.05,
    ):

        self.statistic = statistic
        self.data_transform = data_transform
        self.n_bootstrap_samples = n_bootstrap_samples
        self.n_window_samples = n_window_samples
        self.n_compromised = n_compromised
        self.significance_level = significance_level
        if bootstrap_method == "simple":
            self.bootstrap_method = self._simple_bootstrap
        elif bootstrap_method == "time":
            self.bootstrap_method = self._time_bootstrap
        else:
            raise NotImplemented(f"{bootstrap_method} is not a valid bootstrap_method")

        self.localization_thresholds_ = None
        self.detection_thresholds_ = None
        self.bootstrap_score_distribution_ = None
        return

    def fit(self, X_boot, Y_boot, random_state=None):
        """Sets the detection and localization thresholds using the specified bootstrapping method.
        Parameters
        ----------
        X_boot : array-like, (n_samples, n_features)
            The empirical distribution which will be concatenated with Y_boot to get the simulated null distribution
            which bootstrapping will be performed on.
        Y_boot : array-like, (n_samples, n_features), or None
            If empirical distribution, then it will be concatenated with X_boot to simulate the null distribution, but
            if None, then it is assumed that X_boot already is the simulated null distribution.
        Returns
        ----------
        self, (with detection_thresholds_ and localization_thresholds_ set)
        """
        (
            self.detection_thresholds_,
            self.localization_thresholds_,
            self.bootstrap_score_distribution_,
        ) = self.bootstrap_method(X_boot, Y_boot, random_state)

        return self

    def detect_and_localize(self, X, Y, random_state=None, return_scores=False):
        """Performs distribution shift detection and localization to features
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The empirical distribution of samples from a reference distribution
        Y : array-like, shape (n_samples, n_features)
            An empirical distribution of samples from the query distribution, i.e. the distribution we want to know if
            it has shifted away from the reference distribution
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        return_scores: bool, optional (default=False)
            If return_scores is True, then the scores of each features will be returned. The default is False.
        Returns
        ----------
        detection : int
            If at least one feature's score is above the detection threshold (i.e. if a feature shift has been detected)
            returns 1 if detected and 0 otherwise.
        attacked_features : array (n_compromised,) or None
            If a detection has occurred, then this will return the indices of the features which are predicted to
            have shifted (i.e. returns the estimated attack set), and if a detection has not occurred then  returns None
        """

        self._check_fitted()
        rng = check_random_state(random_state)
        scores = self.statistic.fit(X, Y).score_features(random_state=rng)
        # Testing for detection of a shift
        if np.any(scores > self.detection_thresholds_):
            detection = 1
            # since shift detection, now localize
            # first we normalize the scores, so we grab the score which has shifted the greatest from it's "null"
            bootstrap_score_means = np.nanmean(
                self.bootstrap_score_distribution_, axis=0
            )
            bootstrap_score_std = np.nanstd(self.bootstrap_score_distribution_, axis=0)
            normalized_scores = (scores - bootstrap_score_means) / bootstrap_score_std
            attacked_features = normalized_scores.argsort()[-self.n_compromised :]
        else:
            detection = 0
            attacked_features = None
        if not return_scores:
            return detection, attacked_features
        else:
            return detection, attacked_features, scores

    def _simple_bootstrap(self, X_boot, Y_boot, random_state=None):
        """Performs simple bootstrapping"""
        rng = check_random_state(random_state)
        bootstrap_score_distribution = np.zeros(
            shape=(self.n_bootstrap_samples, X_boot.shape[1])
        )

        if Y_boot is not None:
            # Combining X and Y distribution to approximate the null hypothesis, and bootstrapping on that.
            concatenated_distribution = np.concatenate((X_boot, Y_boot), axis=0)
        else:
            # Since Y_boot is None, we are assuming X_boot holds the concatenated_distribution already
            concatenated_distribution = X_boot.copy()
        for B_idx in range(self.n_bootstrap_samples):
            XY = concatenated_distribution[
                rng.choice(
                    concatenated_distribution.shape[0],
                    size=X_boot.shape[0] + Y_boot.shape[0],
                    replace=True,
                )
            ]
            if self.data_transform is None:
                X = XY[: X_boot.shape[0]]
                Y = XY[X_boot.shape[0] :]
            else:
                XY = self.data_transform(XY)
                if XY.shape[0] % 2 == 1:
                    # Some transforms can make XY have an uneven shape,
                    # and thus X,Y will be different sizes, so this ensures they will be the same size
                    X = XY[: int(XY.shape[0] / 2)]
                    Y = XY[int(XY.shape[0] / 2) : XY.shape[0] - 1]
                else:
                    X, Y = XY[: int(XY.shape[0] / 2)], XY[int(XY.shape[0] / 2) :]

            bootstrap_score_distribution[B_idx] = self.statistic.fit(
                X, Y
            ).score_features(random_state=rng)

        bootstrap_score_distribution = np.sort(bootstrap_score_distribution, axis=0)
        # We use the bonferroni correction for multiple hypothesis by dividing the detection significance level by n_dim
        detection_thresholds = bootstrap_score_distribution[
            int(
                self.n_bootstrap_samples
                * (1 - (self.significance_level / X_boot.shape[1]))
            )
        ]
        localization_thresholds = bootstrap_score_distribution[
            int(self.n_bootstrap_samples * (1 - self.significance_level))
        ]
        return (
            detection_thresholds,
            localization_thresholds,
            bootstrap_score_distribution,
        )

    def _time_bootstrap(self, X_boot, Y_boot, random_state=None):
        """Performs a time aware bootstrapping -- Note: X_boot, Y_boot should be large (note: Y_boot can be none if
        X_boot alone is the training dataset)"""
        rng = check_random_state(random_state)
        bootstrap_score_distribution = np.zeros(
            shape=(self.n_bootstrap_samples, X_boot.shape[1])
        )
        if Y_boot is not None:
            # Combining X and Y distribution to approximate the null hypothesis, and bootstrapping on that.
            concatenated_distribution = np.concatenate((X_boot, Y_boot), axis=0)
        else:  # X_boot is already the concatenated_distriubtion
            concatenated_distribution = X_boot.copy()

        bootstrap_split_range = [
            self.n_window_samples,
            concatenated_distribution.shape[0] - self.n_window_samples,
        ]
        bootstrap_split_idxs = rng.randint(
            *bootstrap_split_range, size=self.n_bootstrap_samples
        )
        for B_idx, bootstrap_split in enumerate(bootstrap_split_idxs):
            XY = concatenated_distribution[
                bootstrap_split
                - self.n_window_samples : bootstrap_split
                + self.n_window_samples
            ]
            if self.data_transform is None:
                X, Y = XY[: self.n_window_samples], XY[self.n_window_samples :]
            else:
                XY = self.data_transform(XY)
                if XY.shape[0] % 2 == 1:
                    # Some transforms can make XY have an uneven shape,
                    # and thus X,Y will be different sizes, so this ensures they will be the same size
                    X = XY[: int(XY.shape[0] / 2)]
                    Y = XY[int(XY.shape[0] / 2) : XY.shape[0] - 1]
                else:
                    X, Y = XY[: self.n_window_samples], XY[self.n_window_samples :]

            bootstrap_score_distribution[B_idx] = self.statistic.fit(
                X, Y
            ).score_features(random_state=rng)

        bootstrap_score_distribution = np.sort(bootstrap_score_distribution, axis=0)
        # We use the bonferroni correction for multiple hypothesis by dividing the detection significance level by n_dim
        detection_thresholds = bootstrap_score_distribution[
            int(
                self.n_bootstrap_samples
                * (1 - self.significance_level / X_boot.shape[1])
            )
        ]
        localization_thresholds = bootstrap_score_distribution[
            int(self.n_bootstrap_samples * (1 - self.significance_level))
        ]
        return (
            detection_thresholds,
            localization_thresholds,
            bootstrap_score_distribution,
        )

    def _check_fitted(self, error_message=None):
        """Checks if the p_hat and q_hat models have been fitted else, returns an error"""
        if self.detection_thresholds_ is not None:
            return True
        else:
            if error_message is None:
                raise ValueError(
                    "The density has not been fitted, please fit the density and try again"
                )
            else:
                raise ValueError(error_message)
