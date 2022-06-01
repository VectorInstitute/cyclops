import numpy as np
import torch
from deep_density_model import SingleGaussianizeStep, TorchUnitHistogram
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianDensity:
    """
    Fits a multivariate Gaussian Density to input data with methods for computing log gradient.
    Parameters
    ----------
    None.
    Attributes
    ----------
    mean_ : array-like, shape (n_features,)
        The mean of the fitted Gaussian.
    covariance_ : array-like, shape (n_features, n_features)
        the covariance of the fitted Gaussian
    density_ : torch.MultivariateNormal object,
        The MultivariateNormal object fitted to the emperical mean and covariance
    """

    def __init__(self):
        self.mean_ = None
        self.covariance_ = None
        self.density_ = None

    def fit(self, X):
        """Fits a multivariate Gaussian Density to the empirical data, X
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The empirical data which we will fit a Gaussian to.
        Returns
        -------
        self : the fitted instance"""

        X = check_array(X)
        self.mean_ = X.mean(axis=0).astype(np.float64)
        self.covariance_ = np.cov(X, rowvar=False) + 1e-5 * np.eye(X.shape[1])
        self.covariance_ = self.covariance_.astype(np.float64)
        self.density_ = MultivariateNormal(
            loc=torch.from_numpy(self.mean_),
            covariance_matrix=torch.from_numpy(self.covariance_),
        )
        return self

    def sample(self, n_samples=1, random_state=None):
        """
        Samples from the fitted Gaussian.
        Parameters
        ----------
        n_samples: int, optional (default=1)
            Number of samples to generate.
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        Returns
        -------
        samples: array-like, shape (n_samples, n_features)
            The random samples from the fitted Gaussian.
        """

        self._check_fitted("The density must be fitted before it can be sampled")
        rng = check_random_state(random_state)
        torch.manual_seed(
            rng.randint(10000)
        )  # sets the torch seed using the rng from numpy
        return self.density_.sample((n_samples,)).numpy()

    def conditional_sample(self, x, feature_idx, n_samples=1, random_state=None):
        """
        Computes the conditional distribution of the jth feature of the density and samples from the conditional
        Parameters
        ----------
        x: array-like, shape (n_features)
            The sample which we are going to be conditioning on. More specifically, we will be condtioning on the value
            of the jth feature in x.
        feature_idx: int
            The index of the feature which we will compute the conditional distribution of (i.e. p(x_j | x_{-j}))
        n_samples: int, optional (default=1)
            The number of sample to sample from the conditional distribution
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        Returns
        -------
        conditional_samples: array-like, shape (n_samples,)
            The samples from the conditional distribution. Note: These are univariate samples since this conditional
            is a univarient distribution.
        """
        self._check_fitted()
        rng = check_random_state(random_state)
        conditional_mean, conditional_var = self._calculate_1d_guassian_conditional(
            x,
            feature_idx,
            joint_mean=self.mean_,
            joint_cov=self.covariance,
            random_state=rng,
        )
        conditional_samples = rng.normal(
            loc=conditional_mean, scale=conditional_var[0], size=n_samples
        )
        return conditional_samples

    def gradient_log_prob(self, X):
        """
        Computes the gradient of the log probability of the provided samples under the fitted Gaussian density
        Parameters
        ----------
        X: arrray-like (n_samples, n_features)
            The samples for which to compute the log-probability.
        Returns
        -------
        gradient-log-probability: array-like (n_samples, n_features)
            The gradient of the log probability of each sample"""

        # TODO: see if we can speed this up and perform the gradient on
        #  all the samples at once rather than one at a time
        self._check_fitted(
            "The density must be fitted before sample probabilities can be taken"
        )
        X = check_array(X, ensure_2d=False, dtype=np.float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        grad_log_probs = np.empty_like(X)
        X = torch.from_numpy(X)
        for sample_idx, sample in enumerate(
            X
        ):  # iterating over each sample to calc the gradient of the log prob.
            sample.requires_grad_(True)
            log_prob = self.density_.log_prob(sample)
            grad_log_probs[sample_idx] = torch.autograd.grad(log_prob, sample)[
                0
            ]  # returns tuple with [0] as grad
        return grad_log_probs

    def log_prob(self, X):
        """
        Calculates the log probability of samples X under the fitted gaussian
        Parameters
        ----------
        X: arrray-like (n_samples, n_features)
            The samples for which to compute the log-probability.
        Returns
        -------
        log-probability: array-like (n_samples, n_features)
            The log probability of each sample"""
        self._check_fitted()
        X = check_array(X, ensure_2d=False, dtype=np.float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = torch.from_numpy(X)
        print(X.shape)
        return self.density_.log_prob(X).numpy()

    @staticmethod
    def _calculate_1d_guassian_conditional(
        x, feature_idx, joint_mean, joint_cov, random_state=None
    ):
        """
        Computes the conditional distribution of the ith feature of the density and samples from the conditional
        ref: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf   page 40.
        Parameters
        ----------
        x: array-like, shape (n_features)
            The sample which we are going to be conditioning on. More specifically, we will be condtioning on the value
            of the jth feature in x.
        feature_idx: int
            The index of the feature which we will compute the conditional distribution of (i.e. p(x_j | x_{-j}))
        joint_mean: array-like, shape (n_features)
            The mean vector of the joint model
        joint_cov: array-like, shape (n_features, n_features)
            The covaraince matrix of the joint model

        Returns
        -------
        conditional_mean: float
            The mean of the univariate conditional distribution
        conditional_variance: float
            The variance of the univariate conditional distribution
        """
        rng = check_random_state(random_state)
        mask = np.ones(len(x), dtype=bool)
        mask[feature_idx] = False
        x_nj = x[mask]
        means = np.array(joint_mean).flatten()

        # making it so that j (feature_idx) is the first column
        # i.e. \Simga = [ [var(x_j), cov(x_j, x_{-j})], [cov(x_{-j}, x_j), cov(x_j, x_j)] ]

        cov_11 = joint_cov[np.ix_(~mask, ~mask)]
        cov_12 = joint_cov[np.ix_(~mask, mask)]
        cov_22 = joint_cov[np.ix_(mask, mask)]
        cov_22_inv = np.linalg.inv(cov_22)

        conditional_mean = means[~mask] + cov_12 @ cov_22_inv @ (x_nj - means[mask])
        conditional_var = cov_11 - cov_12 @ cov_22_inv @ cov_12.T

        return conditional_mean, conditional_var

    def _check_fitted(self, error_message=None):
        if self.density_ is None:
            if error_message is None:
                raise ValueError(
                    "The density has not been fitted, please fit the density and try again"
                )
            else:
                raise ValueError(error_message)
        return True


class DeepDensity:
    """
    Uses iteratize Gaussianization to fit a deep (n_layers) density model
    Parameters
    ----------
    n_layers : int
    Attributes
    ----------
    layers_ : object(s),
        The layers for the Gaussianization fitted to the emperical data"""

    def __init__(self, n_layers=2):
        self.n_layers = n_layers
        self.density_ = None
        self.layers_ = None

    def fit(self, X):
        """Iteratively fits a density to the empirical data, X
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The empirical data which we will fit a Gaussian to.
        Returns
        -------
        self : the fitted instance"""

        check_array(X, dtype=np.float)
        X = torch.from_numpy(X)
        layers = []
        for ii in range(self.n_layers):
            layer = SingleGaussianizeStep(n_bins=10, alpha=1, lam_variance=0)
            X = layer.fit_transform(X)
            layers.append(layer)
        self.layers_ = layers
        self._latent = X  # Just for debugging
        return self

    def sample(self, n_samples=1, random_state=None):
        """
        Samples from the fitted density.
        Parameters
        ----------
        n_samples: int, optional (default=1)
            Number of samples to generate.
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        Returns
        -------
        samples: array-like, shape (n_samples, n_features)
            The random samples from the fitted Gaussian.
        """
        self._check_fitted()
        rng = check_random_state(random_state)
        torch.manual_seed(rng.randint(1000))
        if n_samples == 1:
            ravel = True
        else:
            ravel = False
        # Sample from base
        X = self.layers_[0].standard_normal_.sample([n_samples])
        for layer in np.flip(self.layers_):
            X = layer.inverse(X)
        if ravel:
            X = X.reshape(-1)
        return X.numpy()

    def log_prob(self, X):
        """
        Computes and the log-probability of the samples in X under the fitted density
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The input samples which to compute the log-probability of
        Returns
        -------
        log_probability: array-like, shape (n_samples, n_features)
            The log probability of the samples of X under the fitted density"""
        self._check_fitted()
        # Transform
        log_prob = torch.zeros_like(X[:, 0])
        for layer in self.layers_:
            log_prob_layer, X = layer.log_prob(X, return_latent=True)
            log_prob += log_prob_layer

        # Base distribution probability
        if True:
            log_prob += torch.sum(self.layers_[0].standard_normal_.log_prob(X), dim=1)
        return log_prob

    def gradient_log_prob(self, X):
        """
        Computes the gradient of the log probability of the provided samples under the fitted density
        Parameters
        ----------
        X: arrray-like (n_samples, n_features)
            The samples for which to compute the log-probability.
        Returns
        -------
        gradient-log-probability: array-like (n_samples, n_features)
            The gradient of the log probability of each sample"""

        # TODO: see if we can speed this up and perform the gradient on
        #  all the samples at once rather than one at a time
        self._check_fitted()
        X = check_array(X, ensure_2d=False, dtype=np.float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        grad_log_probs = np.empty_like(X)
        X = torch.from_numpy(X)
        for sample_idx, sample in enumerate(
            X
        ):  # iterating over each sample to calc the gradient of the log prob.
            sample = sample.reshape(1, -1)
            sample.requires_grad_(True)
            log_prob = self.log_prob(sample)
            grad_log_probs[sample_idx] = torch.autograd.grad(log_prob, sample)[
                0
            ]  # returns tuple with [0] as grad
        return grad_log_probs

    def _check_fitted(self, error_message=None):
        if self.layers_ is None:
            if error_message is None:
                raise ValueError(
                    "The density has not been fitted, please fit the density and try again"
                )
            else:
                raise ValueError(error_message)
        return True


class Knn:
    """
    Uses Sklearn's NearestNeighbors class to learn the neighborhood of points in the training data, and samples directly
    from the emperical distribution or can perform an estimated conditional sample, p(x_j | x_{-j}) based on the
    neighborhood around the point of interest (x_{-j}), without taking the feature of interest (x_j) into account.
    Parameters
    ----------
    n_neighbors : int
        The number of neighbors to consider and return for each neighborhood sample.
    Attributes
    ----------
    knn : NearestNeighbors instance
        An instance of the Sklearn NearestNeighbors class
    X_train_ : array-like, shape (n_samples, n_features)
        During the model's fitting the training data is saved so that it can be sampled from later
    """

    def __init__(self, n_neighbors=100):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="kd_tree",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
        self.X_train_ = None

    def fit(self, X):
        """
        Validates and saves the training data.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training data to be used later during neighborhood sampling.
        Returns
        ----------
        self (with training data copied)
        """
        X = check_array(X)
        self.X_train_ = X
        return self

    def sample(self, n_samples, random_state=None, with_replacement=True):
        """Uniformly draws samples from the saved emperical training data
        Parameters
        ----------
        n_samples : int
            The number of samples to be drawn from the training data, If with_replacement is False, then n_samples must
             be less than the number of training samples
        random_state: int, RandomState instance, or None, optional (default=None)
            If int, then the random state is set using np.random.RandomState(int),
            if RandomState instance, then the instance is used directly, if None then a RandomState instance is
            used as if np.random() was called
        with_replacement : bool
            If true, sampling is performed with replacement, and if false then without.
        Returns
        ----------
        Samples : array-like (n_samples, n_features)
            The samples uniformly drawn from the training data
        """
        self._check_fitted()
        rng = check_random_state(random_state)
        sample_idxs = rng.choice(
            self.X_train_.shape[0], size=n_samples, replace=with_replacement
        )
        return self.X_train_[sample_idxs]

    def conditional_sample(self, feature_idx, X):
        """
        Estimates conditional sampling by finding the k-nearest neighbors to x without taking the feature_idx^{th}
        feature into account. In other words, if j=feature_idx, finds the k-nearest neighbors to x_{-j} (x with feature
        j dropped).
        Parameters
        ----------
        feature_idx : int
            The index of the feature we are not conditioning on, i.e. the feature to be dropped when searching for
             the k-nearest neighbors
        X: array-like, shape (n_samples, n_features) or shape (n_features, )
            The sample(s) which will be conditioned upon. If x is a single sample of shape (n_features, ), then x will
            be reshaped to (n_samples, n_features) with n_samples = 1
        Returns
        -------
        neighborhood_conditional_samples : array-like, shape (n_samples, n_neighbors, n_features)
            A 3 dimensional array containing the neighbors for the samples conditioned upon.
            For example, neighborhood_conditional_samples[0] will have k (k=n_neighbors) rows, which correspond to the
            k neighbors we set out to find (k is set during this class initialization) in order of increasing distance
            from the search point (x), and the columns are the feature values for the neighbor.
            Note: if x is included in X_train_, then x will by construction be returned as the first neighbor. If this
            behavior is unwanted, set the n_neighbors = n_neighbors+1, and throw out the first neighbor each time.
        """
        self._check_fitted()
        if len(x.shape) == 1:
            X = x.reshape(1, -1)

        X_train_not_j = np.delete(
            self.X_train_, feature_idx, axis=1
        )  # removes the jth feature from the emperical dist
        X_not_j = np.delete(
            X, feature_idx, axis=1
        )  # removes the jth feature from the query data
        neighbor_idxs = self.knn.fit(X_train_not_j).kneighbors(
            X_not_j, return_distance=False
        )
        neighborhood_conditional_samples = self.X_train_[neighbor_idxs, feature_idx]
        return neighborhood_conditional_samples

    def _check_fitted(self, error_message=None):
        """Checks if the estimator has been fitted."""
        if self.X_train_ is None:
            if error_message is None:
                raise ValueError("Please fit knn and try again")
            else:
                raise ValueError(error_message)
        return True
