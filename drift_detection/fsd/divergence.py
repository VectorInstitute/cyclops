# built in methods
from copy import copy
# external
import numpy as np
from sklearn.utils import check_array, check_random_state
from scipy.stats import ks_2samp as ks_stat

class FisherDivergence:
    """
    A class for computing the conditional Fisher divergence for two densities.
    Parameters
    ----------
    density_model : density-object,
        The density object which will be called to estimate the P and Q densities
    n_expectation: int,
        The number of samples used in estimate the expectation of the divergence of p(x) and q(x)
    Attributes
    ----------
    p_hat_: density-object,
        A copy of the estimator given in density_model, which is then fit on X (the empirical p distribution)
    q_hat: density-object,
        A copy of the estimator given in density_model, which is then fit on Y (the empirical q distribution)
    """
    def __init__(self, density_model, n_expectation=100):
        self.density_model = density_model
        self.n_expectation = n_expectation
        self.p_hat_ = None
        self.q_hat_ = None

    def fit(self, X, Y):
        """
        Fits a density specified by density_model to the reference empirical distribution X and
         the query empirical distribution Y.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The empirical distribution of samples from a reference distribution
        Y : array-like, shape (n_samples, n_features)
            An empirical distribution of samples from the query distribution, i.e. the distribution we want to know if
            it has shifted away from the reference distribution
        Returns
        -------
        self (with self.p_hat_ and self.q_hat_ set)
        """
        X = check_array(X)
        Y = check_array(Y)
        self.p_hat_ = copy(self.density_model).fit(X)  # creates a copy so p_hat is not refit with q_hat
        self.q_hat_ = copy(self.density_model).fit(Y)  # also creates a copy in case base density model is used later
        return self

    def score_features(self, random_state=None):
        """
        Computes the feature-wise divergence using the Fisher Divergence via sampling from both densities and averaging
        over 2*n_expectation times.
        Parameters
        ----------
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        Returns
        -------
        featurewise_divergence: array-like, shape (n_features,)
        """
        self._check_fitted()
        rng = check_random_state(random_state)
        # creating an array of samples from both p_hat and q_hat
        samples = np.concatenate((self.p_hat_.sample(self.n_expectation),
                                 self.q_hat_.sample(self.n_expectation)), axis=0)
        # getting the gradient of the log probability of those samples under p_hat and q_hat
        p_grad_log_prob = self.p_hat_.gradient_log_prob(samples)
        q_grad_log_prob = self.q_hat_.gradient_log_prob(samples)

        feature_divergence = ((p_grad_log_prob - q_grad_log_prob)**2).sum(axis=0)
        return feature_divergence / (self.n_expectation * 2)

    def _check_fitted(self, error_message=None):
        """Checks if the p_hat and q_hat models have been fitted else, returns an error"""
        if self.p_hat_ is not None and self.q_hat_ is not None:
            return True
        else:
            if error_message is None:
                raise ValueError('The density has not been fitted, please fit the density and try again')
            else:
                raise ValueError(error_message)


class ModelKS(FisherDivergence):
    """
    Computes the featurewise Kolmogorov-Smirnov Test between samples from two estimated densities.
    """
    def __init__(self, density_model, n_expectation=100, n_conditional_samples=1000):
        super().__init__(density_model, n_expectation)
        self.n_conditional_samples = n_conditional_samples


    def score_features(self, random_state=None):
        """
        Performs a feature wise K-S two sample test by first sampling n_samples from both fitted densities
        and then performs a K-S test on those two sampled distributions
        Parameters
        ----------
        n_samples: int,
            The number of samples used to create the empirical distribution for each density
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        Returns
        -------
        featurewise_scores: array-like, shape (n_features,)
            A vector of feature divergences
        """
        self._check_fitted()
        rng = check_random_state(random_state)
        samples = np.concatenate((self.p_hat_.sample(self.n_expectation),
                                  self.q_hat_.sample(self.n_expectation)), axis=0)
        running_KS_divergence = np.zeros(shape=(samples.shape[1],))
        for sample in samples:
            for j_to_condition_on in range(sample.shape[0]):
                p_conditional_samples = self.p_hat_.conditional_sample(sample, j_to_condition_on,
                                                                       n_samples=self.n_conditional_samples,
                                                                       random_state=rng)
                q_conditional_samples = self.q_hat_.conditional_sample(sample, j_to_condition_on,
                                                                       n_samples=self.n_conditional_samples,
                                                                       random_state=rng)
                running_KS_divergence[j_to_condition_on] += ks_stat(p_conditional_samples, q_conditional_samples)[0]
        return running_KS_divergence / (2 * self.n_expectation)

class KnnKS:
    """Computes featurewise Kolmogrov Smirnov two sample tests from the conditional neighborhoods of the Knn fit on X
     and Y.
    Parameters
    ----------
    knn_model : knn-object,
        The K nearest neighbors object which will be called to estimate the P and Q densities
    n_expectation : int,
        The number of samples used in estimate the expectation of the divergence of p(x) and q(x)
    Attributes
    ----------
    p_hat_ : density-object,
        A copy of the estimator given in density_model, which is then fit on X (the empirical p distribution)
    q_hat : density-object,
        A copy of the estimator given in density_model, which is then fit on Y (the empirical q distribution)
    n_dims_ : int,
        The number of dimensions in P or Q
    """
    def __init__(self, knn_model, n_expectation=100):
        self.knn_model = knn_model
        self.n_expectation = n_expectation
        self.p_hat_ = None
        self.q_hat_ = None
        self.n_dims_ = None

    def fit(self, X, Y):
        """"Fits the Knn neighborhood for the empirical distributions of p and q
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The empirical distribution of samples from a reference distribution
        Y : array-like, shape (n_samples, n_features)
            An empirical distribution of samples from the query distribution, i.e. the distribution we want to know if
            it has shifted away from the reference distribution
        Returns
        -------
        self (with self.p_hat_ and self.q_hat_ set)
        """
        self.p_hat_ = copy(self.knn_model).fit(X)
        self.q_hat_ = copy(self.knn_model).fit(Y)
        self.n_dims_ = X.shape[1]
        return self

    def score_features(self, random_state=None):
        """Returns featurewise divergence by performing the Kolmogrov Smirnov two sample tests on the neighborhoods
        of samples uniformly drawn from X and Y, with the j^th dimension removed.
        Parameters
        ----------
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        Returns
        -------
        featurewise_scores: array-like, shape (n_features,)
            A vector of feature divergences"""
        self._check_fitted()
        rng = check_random_state(random_state)
        featurewise_KS_stat = np.zeros(shape=(self.n_dims_, ))
        # here we draw samples from the empirical distributions of X and Y
        samples = np.concatenate((self.p_hat_.sample(n_samples=self.n_expectation, random_state=rng),
                                  self.q_hat_.sample(n_samples=self.n_expectation, random_state=rng)))
        for feature_idx in range(self.n_dims_):
            p_neighborhoods = self.p_hat_.conditional_sample(feature_idx, samples)
            q_neighborhoods = self.q_hat_.conditional_sample(feature_idx, samples)
            # neighborhoods are of shape (n_conditional_expectation, n_neighbors)
            # so we loop over the different neighborhoods (n_cond_expect) to approx E[\phi(p(x | x_nj), q(x | x_nj))]
            for p_conditional_neighborhood, q_conditional_neighborhood in zip(p_neighborhoods, q_neighborhoods):
                featurewise_KS_stat[feature_idx] += ks_stat(p_conditional_neighborhood, q_conditional_neighborhood)[0]
        return featurewise_KS_stat / (2*self.n_expectation)

    def _check_fitted(self, error_message=None):
        """Checks if the p_hat and q_hat models have been fitted else, returns an error"""
        if self.p_hat_ is not None and self.q_hat_ is not None:
            return True
        else:
            if error_message is None:
                raise ValueError('The density has not been fitted, please fit the density and try again')
            else:
                raise ValueError(error_message)