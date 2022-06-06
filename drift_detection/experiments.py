import math
import random
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest

def gaussian_noise_subset(X, noise_amt, normalization=1.0, delta_total=1.0, clip=True):
    """Creates gaussian noise of specificed parameters in input data.

    Parameters
    ----------
    X: numpy.matrix
        covariate data
    noise_amt: int
        standard deviation of gaussian noise
    normalization: int
        normalization parameter to divide noise by (e.g. 255 for images)
    delta_total: float
        fraction of data affected

    """
    X_df = pd.DataFrame(X)
    bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
    c_cols = [x for x in X_df.columns if x not in bin_cols]
    indices = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * delta_total), replace=False
    )
    X_mod = X[np.ix_(indices, c_cols)]
    if len(c_cols) == 1:
        noise = np.random.normal(0, noise_amt / normalization, X_mod.shape[0]).reshape(X_mod.shape[0],1)
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
def knockout_shift(X, y, cl, delta):
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


def changepoint_shift(X_s, y_s, X_t, y_t, cl, n_shuffle=0.25, rank=False):
    """changepoint shift swaps features on a changepoint axis.

    Parameters
    ----------
    X_s: numpy.matrix
        source data
    y_s: list
        source label
    X_t: numpy.matrix
        target data
    y_t: list
        target label
    cl: int
        class (e.g. 0,1,2,3, etc.)
    n_shuffle: float
        number of features to shuffle
    rank: Bool
        should features should be ranked or not?

    """
    n_feats = X_s.shape[1]
    n_shuffle_feats = int(n_shuffle * n_feats)

    ## Get importance values - should sub for model-specific
    selector = SelectKBest(k=n_feats)
    selection = selector.fit(X_s, y_s)
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
    for i in range(len(X_t)):
        if y_t[i] == cl:
            X_t[i, :] = X_t[i, shuffle_list]
    return (X_t, y_t)


def multiway_feat_association_shift(
    X_t, y_t, n_shuffle=0.25, keep_rows_constant=True, repermute_each_column=True
):
    """multiway_feat_association_shift swaps individuals within features.

    Parameters
    ----------
    X_t: numpy.matrix
        target data
    y_t: list
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

    n_inds = X_t.shape[0]
    n_shuffle_inds = int(n_shuffle * n_inds)
    shuffle_start = np.random.randint(n_inds - n_shuffle_inds)
    shuffle_end = shuffle_start + n_shuffle_inds
    shuffle_list = np.random.permutation(range(shuffle_start, shuffle_end))
    for i in range(X_t.shape[1]):
        rng = np.random.default_rng(i)
        if repermute_each_column:
            rng.random(1)
            shuffle_start = np.random.randint(n_inds - n_shuffle_inds)
            shuffle_end = shuffle_start + n_shuffle_inds
        if not keep_rows_constant:
            rng.random(1)
            shuffle_list = np.random.permutation(range(shuffle_start, shuffle_end))
        indices = (
            list(range(0, shuffle_start))
            + list(shuffle_list)
            + list(range(shuffle_end, n_inds))
        )
        # Implement so that it changes only for a specific class
        X_t[:, i] = X_t[indices, i]

    return (X_t, y_t)


def binary_noise_subset(X, p, delta_total=1.0):
    """Creates binary noise of specificed parameters in input data.

    Parameters
    ----------
    X: numpy.matrix
        covariate data
    p: float
        proportion of 1s
    delta_total: float
        fraction of data affected

    """
    X_df = pd.DataFrame(X)
    bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
    indices = np.random.choice(
        X.shape[0], math.ceil(X.shape[0] * delta_total), replace=False
    )
    X_mod = X[indices, :][:, bin_cols]
    if X_mod.shape[1] == 1:
        noise = np.random.binomial(1, p, X_mod.shape[0])
    else:
        noise = np.random.binomial(1, p, (X_mod.shape[0], X_mod.shape[1]))
    X[np.ix_(indices, bin_cols)] = noise
    return X, indices

def age_shift(X_s, y_s, X_t, y_t, col="age"):
    raise NotImplementedError
    
def sex_shift(X_s, y_s, X_t, y_t, col="sex"):
    raise NotImplementedError 

def apply_shift(X_s_orig, y_s_orig, X_te_orig, y_te_orig, shift):

    """apply_shift.

    Parameters
    ----------
    X_s_orig: numpy.matrix
        source data
    y_s_orig: list
        source label
    X_te_orig: numpy.matrix
        target data
    y_te_orig: list
        target label
    shift: String
        shift type
    """

    X_te_1 = None
    y_te_1 = None

    if shift == "large_gn_shift_1.0":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 100.0, normalization=1.0, delta_total=1.0, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_gn_shift_1.0":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 10.0, normalization=1.0, delta_total=1.0, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "small_gn_shift_1.0":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 1.0, normalization=1.0, delta_total=1.0, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "large_gn_shift_0.5":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 100.0, normalization=1.0, delta_total=0.5, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_gn_shift_0.5":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 10.0, normalization=1.0, delta_total=0.5, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "small_gn_shift_0.5":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 1.0, normalization=1.0, delta_total=0.5, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "large_gn_shift_0.1":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 100.0, normalization=1.0, delta_total=0.1, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_gn_shift_0.1":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 10.0, normalization=1.0, delta_total=0.1, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "small_gn_shift_0.1":
        X_te_1, _ = gaussian_noise_subset(
            X_te_orig, 1.0, normalization=1.0, delta_total=0.1, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "ko_shift_0.1":
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.1)
    elif shift == "ko_shift_0.5":
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.5)
    elif shift == "ko_shift_1.0":
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 1.0)
    elif shift == "cp_shift_0.75":
        X_te_1, y_te_1 = changepoint_shift(
            X_s_orig, y_s_orig, X_te_orig, y_te_orig, 0, n_shuffle=0.75, rank=True
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "cp_shift_0.25":
        X_te_1, y_te_1 = changepoint_shift(
            X_s_orig, y_s_orig, X_te_orig, y_te_orig, 0, n_shuffle=0.25, rank=True
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.75_krc_rec":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.75,
            keep_rows_constant=True,
            repermute_each_column=True,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.5_krc_rec":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.5,
            keep_rows_constant=True,
            repermute_each_column=True,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.25_krc_rec":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.25,
            keep_rows_constant=True,
            repermute_each_column=True,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.75_krc":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.75,
            keep_rows_constant=True,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.5_krc":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.5,
            keep_rows_constant=True,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.25_krc":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.25,
            keep_rows_constant=True,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.75":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.75,
            keep_rows_constant=False,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.5":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.5,
            keep_rows_constant=False,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.25":
        X_te_1, y_te_1 = multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.25,
            keep_rows_constant=False,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "large_bn_shift_1.0":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.5, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_bn_shift_1.0":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.1, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == "small_bn_shift_1.0":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.01, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == "large_bn_shift_0.5":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.5, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_bn_shift_0.5":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.1, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == "small_bn_shift_0.5":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.01, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == "large_bn_shift_0.1":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.5, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_bn_shift_0.1":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.1, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == "small_bn_shift_0.1":
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.01, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == "age_pediatric":
        X_te_1, _ = age_shift(X_s_orig, y_s_orig, X_te_orig, y_te_orig)
        y_te_1 = y_te_orig.copy()
    return (X_te_1, y_te_1)
