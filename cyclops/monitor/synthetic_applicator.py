"""SyntheticShiftApplicator class."""
import math

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf
from sklearn.feature_selection import SelectKBest

from .utils import get_args


class SyntheticShiftApplicator:
    """The SyntheticShiftApplicator class is used induce synthetic dataset shift.

    Examples
    --------
    >>> from drift_detection.experimenter import Experimenter
    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)
    >>> applicator = SyntheticShiftApplicator(shift_type="gn_shift")
    >>> X_shift, y_shift = applicator.apply_shift(X_train, noise_amt=0.1, delta=0.1)

    Parameters
    ----------
    shift_type: str
        method used to induce shift in data. Options include:
        "gn_shift", "ko_shift", "cp_shift", "mfa_shift", "bn_shift", "tolerance_shift"

    """

    def __init__(self, shift_type: str, **kwargs):

        self.shift_type = shift_type

        self.shift_types = {
            "gn_shift": gaussian_noise_shift,
            "ko_shift": knockout_shift,
            "fs_shift": feature_swap_shift,
            "fa_shift": feature_association_shift,
            "bn_shift": binary_noise_shift,
            "categorical_shift": categorical_shift,
        }
        self.shift_args = get_args(self.shift_types[self.shift_type], kwargs)

        if self.shift_type not in self.shift_types:
            raise ValueError(f"shift_type must be one of {self.shift_types.keys()}")

    def apply_shift(self, X, metadata, metadata_mapping):
        """apply_shift.

        Returns
        -------
        X: numpy.matrix
            Data to have noise added

        """
        # check if X is a numpy array or dataset
        if isinstance(X, np.ndarray):
            X_shift = X.copy()
        y_shift = None

        X_shift, _ = self.shift_types[self.shift_type](
            X_shift, metadata, metadata_mapping, **self.shift_args
        )

        return (X_shift, y_shift)


def categorical_shift(
    X: np.ndarray,
    metadata: pd.DataFrame,
    metadata_mapping: dict,
    categorical_column: str,
    target_category: str,
):
    """Create categorical shift by changing a fraction of samples from a class.

    Parameters
    ----------
    X: numpy.matrix    # import ListConfig from Omegaconf

        covariate data
    metadata: pd.DataFrame
        metadata data
    categorical_column: str
        name of the column to change
    target_category: str
        name of the category to change to

    Returns
    -------
    X: numpy.matrix
        Data to have noise added
    y: numpy.matrix
        placeholder for labels

    """
    y_target = None
    cat_col = metadata_mapping[categorical_column]

    # check if cat_col is a list or ListConfig
    if isinstance(cat_col, (list, ListConfig)):
        if isinstance(cat_col, ListConfig):
            cat_col = OmegaConf.to_object(cat_col)
        name, value = str.split(target_category, ":")
        # check if target_name in cat_col, if not throw error
        if name not in cat_col:
            raise ValueError(f"target_name {name} not in categorical_column {cat_col}")
        metadata.reset_index(drop=True, inplace=True)
        target_indices = metadata.loc[metadata[name] == int(value)].index.values

    elif "-" in target_category:
        lower, upper = str.split(target_category, "-")
        lower, upper = int(lower), int(upper)
        metadata.reset_index(drop=True, inplace=True)
        target_indices = metadata.loc[
            (metadata[cat_col] >= lower) & (metadata[cat_col] <= upper)
        ].index.values
    else:
        metadata.reset_index(drop=True, inplace=True)
        target_indices = metadata.loc[metadata[cat_col] == target_category].index.values

    X_target = X[target_indices]
    return X_target, y_target


# def apply_predefined_shift(
#     predefined_shift: str,
#     X: np.ndarray,
#     y: np.ndarray = None,
#     X_ref: np.ndarray = None,
#     y_ref: np.ndarray = None,
# ):
#     """Apply a predefined shift.

#     Parameters
#     ----------
#     predefined_shift: String
#         Name of shift type to use.
#     X: numpy.matrix
#         Source data.
#     y: list
#         Source label.
#     X_ref: numpy.matrix
#         Target data.
#     y_ref: list
#         Target label.

#     Returns
#     -------
#     X_shift: numpy.matrix
#         shifted features
#     y_shift: numpy.array
#         placeholder for labels

#     """
#     X_shift = X.copy()
#     y_shift = None

#     if predefined_shift == "large_gn_shift_1.0":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift,
#             10.0,
#             normalization=1.0,
#             delta=1.0,
#             clip=False,
#         )
#     elif predefined_shift == "medium_gn_shift_1.0":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 1.0, normalization=1.0, delta=1.0, clip=False
#         )
#     elif predefined_shift == "small_gn_shift_1.0":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 0.1, normalization=1.0, delta=1.0, clip=False
#         )
#     elif predefined_shift == "large_gn_shift_0.5":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 10.0, normalization=1.0, delta=0.5, clip=False
#         )
#     elif predefined_shift == "medium_gn_shift_0.5":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 1.0, normalization=1.0, delta=0.5, clip=False
#         )
#     elif predefined_shift == "small_gn_shift_0.5":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 0.1, normalization=1.0, delta=0.5, clip=False
#         )
#     elif predefined_shift == "large_gn_shift_0.1":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 10.0, normalization=1.0, delta=0.1, clip=False
#         )
#     elif predefined_shift == "medium_gn_shift_0.1":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 1.0, normalization=1.0, delta=0.1, clip=False
#         )
#     elif predefined_shift == "small_gn_shift_0.1":
#         X_shift, _ = gaussian_noise_shift(
#             X_shift, 0.1, normalization=1.0, delta=0.1, clip=False
#         )
#     elif predefined_shift == "ko_shift_0.1":
#         X_shift, y_shift = knockout_shift(X_shift, y, 0, 0)
#     elif predefined_shift == "ko_shift_0.5":
#         X_shift, y_shift = knockout_shift(X_shift, y, 0, 1)
#     elif predefined_shift == "ko_shift_1.0":
#         X_shift, y_shift = knockout_shift(X_shift, y, 0, 2)
#     elif predefined_shift == "cp_shift_0.75":
#         X_shift, y_shift = feature_swap_shift(
#             X_shift, y, X_ref, y_ref, 0, n_shuffle=0.75, rank=True
#         )
#     elif predefined_shift == "cp_shift_0.25":
#         X_shift, y_shift = feature_swap_shift(
#             X_shift, y, X_ref, y_ref, 0, n_shuffle=0.25, rank=True
#         )
#     elif predefined_shift == "mfa_shift_0.75_krc_rec":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.75,
#             keep_rows_constant=True,
#             repermute_each_column=True,
#         )
#     elif predefined_shift == "mfa_shift_0.25_krc_rec":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.25,
#             keep_rows_constant=True,
#             repermute_each_column=True,
#         )
#     elif predefined_shift == "mfa_shift_0.75_krc":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.75,
#             keep_rows_constant=True,
#             repermute_each_column=False,
#         )
#     elif predefined_shift == "mfa_shift_0.5_krc":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.5,
#             keep_rows_constant=True,
#             repermute_each_column=False,
#         )
#     elif predefined_shift == "mfa_shift_0.25_krc":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.25,
#             keep_rows_constant=True,
#             repermute_each_column=False,
#         )
#     elif predefined_shift == "mfa_shift_0.75":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.75,
#             keep_rows_constant=False,
#             repermute_each_column=False,
#         )
#     elif predefined_shift == "mfa_shift_0.5":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.5,
#             keep_rows_constant=False,
#             repermute_each_column=False,
#         )
#     elif predefined_shift == "mfa_shift_0.25":
#         X_shift, y_shift = feature_association_shift(
#             X_shift,
#             y,
#             n_shuffle=0.25,
#             keep_rows_constant=False,
#             repermute_each_column=False,
#         )
#     elif predefined_shift == "large_bn_shift_1.0":
#         X_shift, _ = binary_noise_shift(X_shift, 0.5, 1.0)
#     elif predefined_shift == "medium_bn_shift_1.0":
#         X_shift, _ = binary_noise_shift(X_shift, 0.1, 1.0)
#     elif predefined_shift == "small_bn_shift_1.0":
#         X_shift, _ = binary_noise_shift(X_shift, 0.01, 1.0)
#     elif predefined_shift == "large_bn_shift_0.5":
#         X_shift, _ = binary_noise_shift(X_shift, 0.5, 0.5)
#     elif predefined_shift == "medium_bn_shift_0.5":
#         X_shift, _ = binary_noise_shift(X_shift, 0.1, 0.5)
#     elif predefined_shift == "small_bn_shift_0.5":
#         X_shift, _ = binary_noise_shift(X_shift, 0.01, 0.5)
#     elif predefined_shift == "large_bn_shift_0.1":
#         X_shift, _ = binary_noise_shift(X_shift, 0.5, 0.1)
#     elif predefined_shift == "medium_bn_shift_0.1":
#         X_shift, _ = binary_noise_shift(X_shift, 0.1, 0.1)
#     elif predefined_shift == "small_bn_shift_0.1":
#         X_shift, _ = binary_noise_shift(X_shift, 0.01, 0.1)
#     else:
#         raise ValueError(
#             "Not a pre-defined shift, \
#             specify custom parameters using appropriate function"
#         )
#     return (X_shift, y_shift)


def gaussian_noise_shift(
    X: np.ndarray,
    metadata: pd.DataFrame,
    metadata_mapping: dict,
    noise_amt: float = 0.5,
    normalization: float = 1,
    delta: float = 0.5,
    clip: bool = False,
):
    """Create gaussian noise of specificed parameters in input data.

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

    Returns
    -------
    X: numpy.matrix
        covariate data with gaussian noise
    indices: list
        indices of data affected

    """
    # unused variables
    _, _ = metadata, metadata_mapping

    # add if temporal then flatten then unflatten at end
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
def knockout_shift(
    X: np.ndarray,
    metadata: pd.DataFrame,
    metadata_mapping: dict,
    y: np.ndarray,
    delta: float = 0.5,
    shift_class: int = 1,
):
    """Create class imbalance by removing a fraction of samples from a class.

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

    Returns
    -------
    X: numpy.matrix
        covariate data with class imbalance
    y: numpy.array
        placeholer for labels

    """
    # unused variables
    _, _ = metadata, metadata_mapping

    del_indices = np.where(y == shift_class)[0]
    until_index = math.ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    X = np.delete(X, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    return X, y


def feature_swap_shift(
    X: np.ndarray,
    metadata: pd.DataFrame,
    metadata_mapping: dict,
    y: np.ndarray,
    X_ref: np.ndarray = None,
    y_ref: np.ndarray = None,
    shift_class: int = 1,
    n_shuffle: float = 0.25,
    rank: bool = False,
):
    """Feature swap shift swaps features on a changepoint axis.

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

    Returns
    -------
    X: numpy.matrix
        covariate data with feature swap
    y: numpy.array
        labels for covariate data

    """
    # unused variables
    _, _ = metadata, metadata_mapping

    if isinstance(X_ref, np.ndarray):
        n_feats = X_ref.shape[1]
    n_shuffle_feats = int(n_shuffle * n_feats)

    # Get importance values - should sub for model-specific
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
        if y[i] == shift_class:
            X[i, :] = X[i, shuffle_list]
    return (X, y)


def feature_association_shift(
    X: np.ndarray,
    metadata: pd.DataFrame,
    metadata_mapping: dict,
    y: np.ndarray,
    n_shuffle: float = 0.25,
    keep_rows_constant: bool = True,
    repermute_each_column: bool = True,
):
    """Multiway feature association shift swaps individuals within features.

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

    Returns
    -------
    X: numpy.matrix
        covariate data with feature association
    y: numpy.array
        placeholder for labels

    """
    # unused variables
    _, _ = metadata, metadata_mapping

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


def binary_noise_shift(
    X: np.ndarray,
    metadata: pd.DataFrame,
    metadata_mapping: dict,
    prob: float = 0.5,
    delta: float = 0.5,
):
    """Create binary noise of specificed parameters in input data.

    Parameters
    ----------
    X: numpy.matrix
        covariate data
    p: float
        Proportion of case to control
    delta: float
        fraction of data affected

    Returns
    -------
    X: numpy.matrix
        covariate data with binary noise
    indices: list
        indices of data affected

    """
    # unused variables
    _, _ = metadata, metadata_mapping

    # add if temporal then flatten then unflatten at end
    X_df = pd.DataFrame(X)
    bin_cols = X_df.loc[:, (X_df.isin([0, 1])).all()].columns.values
    indices = np.random.choice(X.shape[0], math.ceil(X.shape[0] * delta), replace=False)
    X_mod = X[indices, :][:, bin_cols]

    if X_mod.shape[1] == 1:
        noise = np.random.binomial(1, prob, X_mod.shape[0])
    else:
        noise = np.random.binomial(1, prob, (X_mod.shape[0], X_mod.shape[1]))

    X[np.ix_(indices, bin_cols)] = noise

    return X, indices
