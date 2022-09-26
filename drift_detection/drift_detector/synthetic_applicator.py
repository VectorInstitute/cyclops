import math
import random
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest


class SyntheticShiftApplicator:

    """
    The SyntheticShiftApplicator class is used induce synthetic dataset shift.
    --------
    >>> from drift_detection.experimenter import Experimenter
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    >>> applicator = SyntheticShiftApplicator()
    >>> X_shift, y_shift = applicator.apply_shift(X_train, X_test, y_train, y_test, "small_gn_shift_0.1")

    Parameters
    ----------
    shift_method: str
        method used to induce shift in data. Options include: "gn_shift", "ko_shift", "cp_shift", "mfa_shift", "bn_shift",
        "tolerance_shift"

    """

    def __init__(self, shift_method: str, **kwargs):

        self.shift_methods = {
            "gn_shift": gaussian_noise_shift,
            "ko_shift": knockout_shift,
            "cp_shift": changepoint_shift,
            "mfa_shift": multiway_feat_association_shift,
            "bn_shift": binary_noise_shift,
            "tolerance_shift": tolerance_shift,
        }

        if self.shift_method not in self.shift_methods.keys():
            raise ValueError(
                "Shift not supported, must be one of: {}".format(
                    self.shift_methods.keys()
                )
            )

    def apply_shift(self, X, **kwargs):

        """apply_shift.

        Returns
        ----------
        X: numpy.matrix
            Data to have noise added.
        """

        X_shift = None
        y_shift = None

        X_shift, _ = self.shift_methods[self.shift_method](
            X, **get_args(self.shift_methods[self.shift_method], kwargs)
        )
        y_shift = y_shift.copy()

        return (X_shift, y_shift)
    


def gaussian_noise_shift(self, X, noise_amt=0.5, delta=0.5, clip=True):
    """Creates gaussian noise of specificed parameters in input data.

    Parameters
    ----------
    X: numpy.matrix
        covariate data
    noise_amt: int
        standard deviation of gaussian noise
    normalization: int
        normalization parameter to divide noise by (e.g. 255 for images)
    delta: float
        fraction of data affected

    """

    ## add if temporal then flatten then unflatten at end
    X_df = pd.DataFrame(X)

    bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
    c_cols = [x for x in X_df.columns if x not in bin_cols]
    indices = np.random.choice(X.shape[0], math.ceil(X.shape[0] * delta), replace=False)
    X_mod = X[np.ix_(indices, c_cols)]

    if len(c_cols) == 1:
        noise = np.random.normal(0, noise_amt / normalization, X_mod.shape[0]).reshape(
            X_mod.shape[0], 1
        )
    else:
        noise = np.random.normal(
            0, noise_amt / normalization, (X_mod.shape[0], len(c_cols))
        )
    if clip:
        X_mod = np.clip(X_mod + noise, 0.0, 1.0)
    else:
        X_mod = X_mod + noise

    X[np.ix_(indices, c_cols)] = X_mod

    return X, indices


# Remove instances of a single class.
def knockout_shift(self, X, y, delta=0.5, cl=1):
    """Creates class imbalance by removing a fraction of samples from a class.

    Parameters
    ----------
    X: numpy.matrix
        covariate data
    y: list
        label data
    cl: int
        class (e.g. 0,1,2,3, etc.)
    delta: float
        fraction of samples removed

    """
    del_indices = np.where(y == cl)[0]
    until_index = math.ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    X = np.delete(X, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    return X, y


def changepoint_shift(self, X, y, X_ref, y_ref, cl=1, n_shuffle=0.25, rank=False):
    """changepoint shift swaps features on a changepoint axis.

    Parameters
    ----------
    X_ref: numpy.matrix
        source data
    y_ref: list
        source label
    X: numpy.matrix
        target data
    y: list
        target label
    cl: int
        class (e.g. 0,1,2,3, etc.)
    n_shuffle: float
        number of features to shuffle
    rank: Bool
        should features should be ranked or not?

    """
    n_feats = X_ref.shape[1]
    n_shuffle_feats = int(n_shuffle * n_feats)

    ## Get importance values - should sub for model-specific
    selector = SelectKBest(k=n_feats)
    selection = selector.fit(X_ref, y_ref)
    ranked_x = sorted(
        zip(selection.scores_, selection.get_support(indices=True)), reverse=True
    )
    shuffle_list = list(range(0, n_feats))

    # Get shuffled features
    if rank:
        prev = shuffle_list[ranked_x[0][1]]
        for i in range(n_shuffle_feats):
            shuffle_list[ranked_x[i][1]] = shuffle_list[
                ranked_x[(i + 1) % n_shuffle_feats][1]
            ]
        shuffle_list[ranked_x[i][1]] = prev
    else:
        prev = shuffle_list[ranked_x[n_feats - n_shuffle_feats][1]]
        for i in range(n_feats - n_shuffle_feats, n_feats):
            shuffle_list[ranked_x[i][1]] = shuffle_list[ranked_x[(i + 1) % n_feats][1]]
        shuffle_list[ranked_x[i][1]] = prev

    # Shuffle features
    for i in range(len(X)):
        if y[i] == cl:
            X[i, :] = X[i, shuffle_list]
    return (X, y)


def multiway_feat_association_shift(
    X, y, n_shuffle=0.25, keep_rows_constant=True, repermute_each_column=True
):
    """multiway_feat_association_shift swaps individuals within features.

    Parameters
    ----------
    X: numpy.matrix
        target data
    y: list
        target label
    cl: int
        class (e.g. 0,1,2,3, etc.)
    n_shuffle: floatnumpy.matrix
        number of individuals to shuffle
    keep_rows_constant:
        are the permutations the same across features?
    repermute_each_column:
        are the individuals selected for permutation the same across features?

    """

    n_inds = X.shape[0]
    n_shuffle_inds = int(n_shuffle * n_inds)
    shuffle_start = np.random.randint(n_inds - n_shuffle_inds)
    shuffle_end = shuffle_start + n_shuffle_inds
    shuffle_list = np.random.permutation(range(shuffle_start, shuffle_end))
    for i in range(X.shape[1]):
        rng = np.random.default_rng(i)
        rng.random(1)
        if repermute_each_column:
            shuffle_start = np.random.randint(n_inds - n_shuffle_inds)
            shuffle_end = shuffle_start + n_shuffle_inds
        if not keep_rows_constant:
            shuffle_list = np.random.permutation(range(shuffle_start, shuffle_end))
        indices = (
            list(range(0, shuffle_start))
            + list(shuffle_list)
            + list(range(shuffle_end, n_inds))
        )
        # Implement so that it changes only for a specific class
        X[:, i] = X[indices, i]

    return (X, y)


def binary_noise_shift(X, p=0.5, delta=0.5):
    """Creates binary noise of specificed parameters in input data.

    Parameters
    ----------
    X: numpy.matrix
        covariate data
    p: float
        Proportion of case to control
    delta: float
        fraction of data affected

    """
    ## add if temporal then flatten then unflatten at end
    X_df = pd.DataFrame(X)
    bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
    indices = np.random.choice(X.shape[0], math.ceil(X.shape[0] * delta), replace=False)
    X_mod = X[indices, :][:, bin_cols]

    if X_mod.shape[1] == 1:
        noise = np.random.binomial(1, p, X_mod.shape[0])
    else:
        noise = np.random.binomial(1, p, (X_mod.shape[0], X_mod.shape[1]))

    X[np.ix_(indices, bin_cols)] = noise

    return X, indices


def tolerance_shift(X, y, X_ref, y_ref, tol_var="gender"):
    raise NotImplementedError
