import math
import random
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest

class Experimenter:
    
    """
    The Experimenter class is used induce dataset shift of various kinds.
    --------
    >>> from drift_detection.experimenter import Experimenter
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    >>> experimenter = Experimenter()
    >>> X_shift, y_shift = experimenter.apply_shift(X_train, X_test, y_train, y_test, "small_gn_shift_0.1")
    Arguments
    ---------
    
    """   
    def __init__(self, X, *,y, noise_amt=None, normalization=1.0, delta=1.0, clip=True, cl=1,):
        kwargs = locals()
        args = [kwargs['backend'], kwargs['p_val'], kwargs['verbose'], kwargs['train_kwargs']]
        
    def transform_to_2d(self, X):
        num_encounters = X.shape[0]
        num_timesteps = X.shape[1]
        
        if X.ndim > 2:
            return X.reshape(num_encounters, -1)

        return X
    
    def transform_to_3d(self, X):
        num_encounters = X.shape[0]
        num_timesteps = X.shape[1]
        
        X = X.reshape(num_encounters,num_timesteps, -1)
        
        return X

    def gaussian_noise_subset(self, X, noise_amt, delta, clip=True):
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

        X_df = pd.DataFrame(self.transform_to_2d(X))

        bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
        c_cols = [x for x in X_df.columns if x not in bin_cols]
        indices = np.random.choice(
            X.shape[0], math.ceil(X.shape[0] * delta), replace=False
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

        if num_timesteps > 1:
            X = X.reshape(num_encounters,num_timesteps, -1)

        return X, indices

    # Remove instances of a single class.
    def knockout_shift(self, X, y, delta, cl=1):
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


    def changepoint_shift(self, X_ref, y_ref, X, y, cl=1, n_shuffle=0.25, rank=False):
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

        return (X_t, y_t)


    def binary_noise_subset(X, p, delta):
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

        X_df = pd.DataFrame(self.transform_to_2d(X))
        bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
        indices = np.random.choice(
            X.shape[0], math.ceil(X.shape[0] * delta), replace=False
        )
        X_mod = X[indices, :][:, bin_cols]

        if X_mod.shape[1] == 1:
            noise = np.random.binomial(1, p, X_mod.shape[0])
        else:
            noise = np.random.binomial(1, p, (X_mod.shape[0], X_mod.shape[1]))

        X[np.ix_(indices, bin_cols)] = noise

        
        return X, indices

    def tolerance_shift(X_ref, y_ref, X, y, tol_var="gender"):
        raise NotImplementedError

    def apply_shift(X_s_orig, y_s_orig, X_te_orig, y_te_orig, shift, tol_var="gender"):

        """apply_shift.

        Parameters
        ----------
        X_s_orig: numpy.matrix
            Source data.
        y_s_orig: list
            Source label.
        X_te_orig: numpy.matrix
            Target data.
        y_te_orig: list
            Target label.
        shift: String
            Name of shift type to use.
        tol_var: String
            Column name of variable to perturb.
        """

        X_te_1 = None
        y_te_1 = None

        if shift == "large_gn_shift_1.0":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 10.0, normalization=1.0, delta=1.0, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "medium_gn_shift_1.0":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 1.0, normalization=1.0, delta=1.0, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "small_gn_shift_1.0":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 0.1, normalization=1.0, delta=1.0, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "large_gn_shift_0.5":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 10.0, normalization=1.0, delta=0.5, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "medium_gn_shift_0.5":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 1.0, normalization=1.0, delta=0.5, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "small_gn_shift_0.5":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 0.1, normalization=1.0, delta=0.5, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "large_gn_shift_0.1":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 10.0, normalization=1.0, delta=0.1, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "medium_gn_shift_0.1":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 1.0, normalization=1.0, delta=0.1, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "small_gn_shift_0.1":
            X_te_1, _ = self.gaussian_noise_subset(
                X_te_orig, 0.1, normalization=1.0, delta=0.1, clip=False
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "ko_shift_0.1":
            X_te_1, y_te_1 = self.knockout_shift(X_te_orig, y_te_orig, 0, 0.1)
        elif shift == "ko_shift_0.5":
            X_te_1, y_te_1 = self.knockout_shift(X_te_orig, y_te_orig, 0, 0.5)
        elif shift == "ko_shift_1.0":
            X_te_1, y_te_1 = self.knockout_shift(X_te_orig, y_te_orig, 0, 1.0)
        elif shift == "cp_shift_0.75":
            X_te_1, y_te_1 = self.changepoint_shift(
                X_s_orig, y_s_orig, X_te_orig, y_te_orig, 0, n_shuffle=0.75, rank=True
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "cp_shift_0.25":
            X_te_1, y_te_1 = self.changepoint_shift(
                X_s_orig, y_s_orig, X_te_orig, y_te_orig, 0, n_shuffle=0.25, rank=True
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.75_krc_rec":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.75,
                keep_rows_constant=True,
                repermute_each_column=True,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.5_krc_rec":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.5,
                keep_rows_constant=True,
                repermute_each_column=True,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.25_krc_rec":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.25,
                keep_rows_constant=True,
                repermute_each_column=True,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.75_krc":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.75,
                keep_rows_constant=True,
                repermute_each_column=False,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.5_krc":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.5,
                keep_rows_constant=True,
                repermute_each_column=False,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.25_krc":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.25,
                keep_rows_constant=True,
                repermute_each_column=False,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.75":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.75,
                keep_rows_constant=False,
                repermute_each_column=False,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.5":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.5,
                keep_rows_constant=False,
                repermute_each_column=False,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "mfa_shift_0.25":
            X_te_1, y_te_1 = self.multiway_feat_association_shift(
                X_te_orig,
                y_te_orig,
                n_shuffle=0.25,
                keep_rows_constant=False,
                repermute_each_column=False,
            )
            y_te_1 = y_te_orig.copy()
        elif shift == "large_bn_shift_1.0":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.5, 1.0)
            y_te_1 = y_te_orig.copy()
        elif shift == "medium_bn_shift_1.0":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.1, 1.0)
            y_te_1 = y_te_orig.copy()
        elif shift == "small_bn_shift_1.0":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.01, 1.0)
            y_te_1 = y_te_orig.copy()
        elif shift == "large_bn_shift_0.5":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.5, 0.5)
            y_te_1 = y_te_orig.copy()
        elif shift == "medium_bn_shift_0.5":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.1, 0.5)
            y_te_1 = y_te_orig.copy()
        elif shift == "small_bn_shift_0.5":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.01, 0.5)
            y_te_1 = y_te_orig.copy()
        elif shift == "large_bn_shift_0.1":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.5, 0.1)
            y_te_1 = y_te_orig.copy()
        elif shift == "medium_bn_shift_0.1":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.1, 0.1)
            y_te_1 = y_te_orig.copy()
        elif shift == "small_bn_shift_0.1":
            X_te_1, _ = self.binary_noise_subset(X_te_orig, 0.01, 0.1)
            y_te_1 = y_te_orig.copy()
        elif shift == "tolerance":
            X_te_1, _ = self.tolerance(X_s_orig, y_s_orig, X_te_orig, y_te_orig, tol_var)
            y_te_1 = y_te_orig.copy()
        else:
            raise ValueError("Not a pre-defined shift, specify custom parameters using appropriate function")
        return (X_te_1, y_te_1)
