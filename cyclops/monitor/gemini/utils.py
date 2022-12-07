"""Utilities for loading and preprocessing gemini data."""
import datetime
import importlib
import random
import types

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# unable to detect unidentified names, star import
# from .constants import *
# from .query import ENCOUNTER_ID


def get_use_case_params(dataset: str, use_case: str) -> types.ModuleType:
    """Import parameters specific to each use-case.

    Parameters
    ----------
    dataset: str
        Name of the dataset, e.g. mimiciv.
    use_case: str
        Name of the use-case, e.g. mortality_decompensation.

    Returns
    -------
    types.ModuleType
        Imported constants module with use-case parameters.

    """
    return importlib.import_module(
        ".".join(["drift_detection", dataset, use_case, "constants"])
    )


def unison_shuffled_copies(array_a, array_b):
    """Shuffle two arrays in unison."""
    assert len(array_a) == len(array_b)
    perm = np.random.permutation(len(array_a))
    return array_a[perm], array_b[perm]


def random_shuffle_and_split(x_train, y_train, x_test, y_test, split_index):
    """Randomly shuffle and split data into train and test sets."""
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = unison_shuffled_copies(x, y)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def get_label(admin_data, X, label="mortality", encounter_id="encounter_id"):
    """Get label from admin data."""
    admin_data = admin_data.drop_duplicates(encounter_id)
    X_admin_data = admin_data[
        admin_data[encounter_id].isin(X.index.get_level_values(0))
    ]
    X_admin_data = (
        X_admin_data.set_index(encounter_id)
        .reindex(list(X.index.get_level_values(0).unique()))
        .reset_index()
    )
    y = X_admin_data[label].astype(int)
    return y


def reshape_2d_to_3d(data, num_timesteps):
    """Reshape 2D data to 3D data."""
    data = data.unstack()
    num_encounters = data.shape[0]
    data = data.values.reshape((num_encounters, num_timesteps, -1))
    return data


def flatten(X):
    """Flatten 3D data to 2D data."""
    X_flattened = X.unstack(1).dropna().to_numpy()
    return X_flattened


def temporal_mean(X):
    """Get temporal mean of data."""
    X_mean = X.groupby(level=[0]).mean()
    return X_mean


def temporal_first(X, y=None):
    """Get temporal first of data."""
    y_first = None
    X_first = X.groupby(level=[0]).first()
    if y is not None:
        y_first = y[:, 0]
    return X_first, y_first


def temporal_last(X, y):
    """Get temporal last of data."""
    X_last = X.groupby(level=[0]).last()
    num_timesteps = y.shape[1]
    y_last = y[:, (num_timesteps - 1)]
    return X_last, y_last


def get_numerical_cols(X: pd.DataFrame):
    """Get numerical columns of temporal dataframe."""
    numerical_cols = [
        col for col in X if not np.isin(X[col].dropna().unique(), [0, 1]).all()
    ]
    return numerical_cols


def scale(X: pd.DataFrame):
    """Scale columns of temporal dataframe.

    Returns
    -------
    model: torch.nn.Module
        feed forward neural network model.

    """
    numerical_cols = get_numerical_cols(X)
    for col in numerical_cols:
        scaler = StandardScaler().fit(X[col].values.reshape(-1, 1))
        X[col] = pd.Series(
            np.squeeze(scaler.transform(X[col].values.reshape(-1, 1))),
            index=X[col].index,
        )
    return X


def normalize(X, aggregation_type):
    """Normalize data."""
    y = None

    if aggregation_type == "mean":
        X_normalized = temporal_mean(X)
    elif aggregation_type == "first":
        (
            X_normalized,
            y,
        ) = temporal_first(X, y)
    elif aggregation_type == "last":
        (
            X_normalized,
            y,
        ) = temporal_last(X, y)
    elif aggregation_type == "time_flatten":
        X_normalized = X.copy()
    elif aggregation_type == "time":
        X_normalized = X.copy()
    else:
        raise ValueError("Incorrect Aggregation Type")
    return X_normalized


def process(X, aggregation_type, timesteps):
    """Process data."""
    if aggregation_type == "time_flatten":
        X_preprocessed = flatten(X)
    elif aggregation_type == "time":
        X_preprocessed = reshape_2d_to_3d(X, timesteps)
    else:
        X_preprocessed = X.dropna().to_numpy()
    return X_preprocessed


def get_dataset_hospital(
    admin_data, x, y, dataset, hospitals, encounter_id="encounter_id", train_frac=0.8
):
    """Get dataset for hospital."""
    # filter hospital
    admin_data = admin_data.loc[admin_data["hospital_id"].isin(hospitals)]
    encounter_ids = list(x.index.get_level_values(0).unique())
    x = x[np.in1d(x.index.get_level_values(0), admin_data[encounter_id])]

    # get source and target data
    x_s = None
    y_s = None
    x_t = None
    y_t = None

    # get experimental dataset
    if dataset == "covid":

        ids_source = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2019, 1, 1))
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 2, 1)),
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2020, 3, 1))
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 8, 1)),
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_winter":
        ids_source = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.month.isin([3, 4, 5, 6, 7, 8, 9, 10]))),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.month.isin([11, 12, 1, 2]))),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_summer":
        ids_source = admin_data.loc[
            (
                (
                    admin_data["admit_timestamp"].dt.month.isin(
                        [1, 2, 3, 4, 5, 10, 11, 12]
                    )
                )
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.month.isin([6, 7, 8, 9]))),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_academic":
        ids_source = admin_data.loc[
            (
                (
                    admin_data["hospital_id"].isin(
                        ["SMH", "MSH", "UHNTG", "UHNTW", "PMH", "SBK"]
                    )
                )
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["hospital_id"].isin(["THPC", "THPM"]))),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_community":
        ids_source = admin_data.loc[
            ((admin_data["hospital_id"].isin(["THPC", "THPM"]))),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["hospital_id"].isin(["SMH", "MSH", "UHNTG", "UHNTW", "PMH"]))),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "simulated_deployment":
        ids_source = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2011, 4, 1))
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2019, 1, 1)),
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2019, 1, 1))
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 8, 1)),
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "random":
        num_train = int(train_frac * len(encounter_ids))
        ids_source = encounter_ids[0:num_train]
        ids_target = encounter_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "covid_baseline":
        dataset_ids = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2019, 1, 1))
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 2, 1)),
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac * len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_summer_baseline":
        ids_source = admin_data.loc[
            (
                (
                    admin_data["admit_timestamp"].dt.month.isin(
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                    )
                )
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.month.isin([6, 7, 8, 9]))),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_winter_baseline":
        ids_source = admin_data.loc[
            (
                (
                    admin_data["admit_timestamp"].dt.month.isin(
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                    )
                )
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.month.isin([11, 12, 1, 2]))),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_academic_baseline":
        dataset_ids = admin_data.loc[
            (
                (
                    admin_data["hospital_id"].isin(
                        ["SMH", "MSH", "UHNTG", "UHNTW", "PMH", "SBK"]
                    )
                )
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac * len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_community_baseline":
        dataset_ids = admin_data.loc[
            ((admin_data["hospital_id"].isin(["THPC", "THPM"]))),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac * len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    y_s = y[np.in1d(encounter_ids, x_s.index.get_level_values(0).unique())]
    y_t = y[np.in1d(encounter_ids, x_t.index.get_level_values(0).unique())]

    assert len(x_s.index.get_level_values(0).unique()) == len(y_s)
    assert len(x_t.index.get_level_values(0).unique()) == len(y_t)

    return (x_s, y_s, x_t, y_t, x_s.columns, admin_data)


def import_dataset_hospital(
    admin_data,
    x,
    y,
    dataset,
    hospital,
    encounter_id,
    seed=1,
    shuffle=True,
    train_frac=0.8,
):
    """Import dataset for hospital-level analysis."""
    # get source and target data
    x_source, y_source, x_test, y_test, feats, admin_data = get_dataset_hospital(
        admin_data, x, y, dataset, hospital, encounter_id
    )

    # get train, validation and test set
    encounter_ids = list(x_source.index.get_level_values(0).unique())

    if shuffle:
        random.Random(seed).shuffle(encounter_ids)

    num_train = int(train_frac * len(encounter_ids))
    train_ids = encounter_ids[0:num_train]
    val_ids = encounter_ids[num_train:]
    x_train, x_val = [
        x_source[np.in1d(x_source.index.get_level_values(0), ids)]
        for ids in [train_ids, val_ids]
    ]
    y_train, y_val = [
        y_source[np.in1d(encounter_ids, ids)] for ids in [train_ids, val_ids]
    ]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), feats, admin_data
