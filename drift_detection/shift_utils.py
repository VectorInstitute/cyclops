from shift_detector import *
from shift_reductor import *
from shift_tester import *
from shift_experiments import *

import re
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sqlalchemy
from sqlalchemy import func, select, desc
from sqlalchemy.sql.expression import and_, or_
from shift_constants import *

import sys
from functools import reduce
import datetime
import joblib
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.metrics import (
    auc,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import select, func, extract, desc
from sqlalchemy.sql.expression import and_

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataQualityTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection

import cyclops.config
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    HOSPITAL_ID,
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    DISCHARGE_DISPOSITION,
    READMISSION,
    AGE,
    SEX,
    TOTAL_COST,
    CITY,
    PROVINCE,
    COUNTRY,
    LANGUAGE,
    LENGTH_OF_STAY_IN_ER,
    REFERENCE_RANGE,
)
from cyclops.processor import featurize
from cyclops.processors.aggregate import Aggregator


def get_scaler(scaler):
    """Get scaler.

    Parameters
    ----------
    scaler: string
        String indicating which scaler to retrieve.

    """
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()


def get_data(hospital, na_cutoff):

    EXTRACT_SAVE_PATH = "/mnt/nfs/project/delirium/drift_exp"

    # load admin
    admin_data = pd.read_parquet(
        os.path.join(EXTRACT_SAVE_PATH, "admin_er_2018_2020.gzip")
    )

    admin_columns = [
        AGE,
        SEX,
        HOSPITAL_ID,
        ENCOUNTER_ID,
        ADMIT_TIMESTAMP,
        DISCHARGE_TIMESTAMP,
        LENGTH_OF_STAY_IN_ER,
    ]
    admin_data = admin_data[admin_columns]

    hosp_ids = admin_data.loc[admin_data["hospital_id"].isin(hospital), "encounter_id"]
    hosp_label = "_".join(sorted(hospital, key=str.lower))
    file_name = os.path.join(EXTRACT_SAVE_PATH, "2018_2020_features.gzip")

    if os.path.exists(file_name):
        x = pd.read_parquet(file_name)
    else:
        # load labs
        labs = pd.read_parquet(os.path.join(EXTRACT_SAVE_PATH, "labs_2018_2020.gzip"))

        # load vitals
        vitals = pd.read_parquet(
            os.path.join(EXTRACT_SAVE_PATH, "vitals_2018_2020.gzip")
        )

        feature_handler = featurize(
            static_data=[admin_data],
            temporal_data=[labs, vitals],
            aggregator=Aggregator(bucket_size=6, window=6),
            reference_cols=[HOSPITAL_ID, ADMIT_TIMESTAMP],
        )

        # Merge static and temporal features (temporal here is actually aggregated into a single bucket!)
        static_features = feature_handler.features["static"]
        temporal_features = feature_handler.features["temporal"]
        merged_features = static_features.join(temporal_features)

        x = merged_features
        x.to_parquet(file_name)

    # Filter based on Hospital
    x = x.loc[x.index.isin(hosp_ids)]

    # features to create indicator features for temporal features.
    ind_cols = [col for col in x if col not in admin_columns]

    # Create indicator features
    x = get_indicators(x, ind_cols)

    return admin_data, x


def get_dataset_hospital(datset, outcome, hospital, na_cutoff):

    # get all data
    (admin_data, x) = get_data(hospital, na_cutoff)

    # process features
    x = remove_missing_feats(x, na_cutoff)

    # only get encounters with length of stay in er
    if outcome == "length_of_stay_in_er":
        x = x[x[outcome].notna()]
        x[outcome] = np.where(x[outcome] > 7, 1, 0)

    # get source and target data
    x_s = None
    y_s = None
    x_t = None
    y_t = None

    # get experimental dataset
    if datset == "covid":
        ids_precovid = admin_data.loc[
            admin_data["admit_timestamp"].dt.date < datetime.date(2020, 3, 1),
            "encounter_id",
        ]
        ids_covid = admin_data.loc[
            admin_data["admit_timestamp"].dt.date > datetime.date(2020, 2, 28),
            "encounter_id",
        ]
        x_s = x.loc[x.index.isin(ids_precovid)]
        x_t = x.loc[x.index.isin(ids_covid)]

    elif datset == "pre-covid":
        ids_precovid = admin_data.loc[
            admin_data["admit_timestamp"].dt.date < datetime.date(2020, 3, 1),
            "encounter_id",
        ]
        x = x.loc[x.index.isin(ids_precovid)]
        x_spl = np.array_split(x, 2)
        x_s = x_spl[0]
        x_t = x_spl[1]

    elif datset == "seasonal":
        ids_summer = admin_data.loc[
            (
                admin_data["admit_timestamp"].dt.month.isin([6, 7, 8])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
            ),
            "encounter_id",
        ]
        ids_winter = admin_data.loc[
            (
                admin_data["admit_timestamp"].dt.month.isin([12, 1, 2])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.isin(ids_winter)]
        x_t = x.loc[x.index.isin(ids_summer)]

    elif datset == "summer":
        ids_summer = admin_data.loc[
            (
                admin_data["admit_timestamp"].dt.month.isin([6, 7, 8])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.isin(ids_summer)]
        x_spl = np.array_split(x, 2)
        x_s = x_spl[0]
        x_t = x_spl[1]

    elif datset == "winter":
        ids_winter = admin_data.loc[
            (
                admin_data["admit_timestamp"].dt.month.isin([12, 1, 2])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.isin(ids_winter)]
        x_spl = np.array_split(x, 2)
        x_s = x_spl[0]
        x_t = x_spl[1]

    elif datset == "hosp_type":
        ids_community = admin_data.loc[
            (
                admin_data["hospital_id"].isin(["THPC", "THPM"])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
                & admin_data["admit_timestamp"].dt.month.isin([6, 7, 8])
            ),
            "encounter_id",
        ]
        ids_academic = admin_data.loc[
            (
                admin_data["hospital_id"].isin(["SMH", "MSH", "UHNTG", "UHNTW"])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
                & admin_data["admit_timestamp"].dt.month.isin([6, 7, 8])
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.isin(ids_community)]
        x_t = x.loc[x.index.isin(ids_academic)]

    elif datset == "academic":
        ids_academic = admin_data.loc[
            (
                admin_data["hospital_id"].isin(["SMH", "MSH", "UHNTG", "UHNTW"])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
                & admin_data["admit_timestamp"].dt.month.isin([6, 7, 8])
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.isin(ids_academic)]
        x_spl = np.array_split(x, 2)
        x_s = x_spl[0]
        x_t = x_spl[1]

    elif datset == "community":
        ids_community = admin_data.loc[
            (
                admin_data["hospital_id"].isin(["THPC", "THPM"])
                & admin_data["admit_timestamp"].dt.year.isin([2018, 2019])
                & admin_data["admit_timestamp"].dt.month.isin([6, 7, 8])
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.isin(ids_community)]
        x_spl = np.array_split(x, 2)
        x_s = x_spl[0]
        x_t = x_spl[1]

    elif datset == "baseline":
        ids_baseline = admin_data.loc[
            (
                admin_data["admit_timestamp"].dt.month.isin([3, 4, 5, 6, 7, 8])
                & admin_data["admit_timestamp"].dt.year.isin([2019])
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.isin(ids_baseline)]
        x_spl = np.array_split(x, 2)
        x_s = x_spl[0]
        x_t = x_spl[1]

    # outcome can be "mortality_in_hospital" or "length_of_stay_in_er"
    y_s = x_s[outcome]
    y_t = x_t[outcome]

    # outcomes = ["mortality_in_hospital", "length_of_stay_in_er"]
    outcomes = ["length_of_stay_in_er"]
    x_s = x_s.drop(columns=outcomes)
    x_t = x_t.drop(columns=outcomes)

    return (x_s, y_s, x_t, y_t, x_s.columns)


def get_indicators(x, ind_cols):
    x_ind = x[ind_cols].isnull().astype(int).add_suffix("_indicator")
    X = pd.concat([x, x_ind], axis=1)
    return X


def remove_missing_feats(x, na_cutoff):
    # drop rows with more than a quarter of features missing
    thres = x.shape[1] * 0.25
    x = x[x.isnull().sum(axis=1) < thres]

    # get number of nas per feature
    feat = x.isnull().sum(axis=0).sort_values(ascending=False)

    # remove nas based on cutoff
    feat_cutoff = x.shape[0] * na_cutoff
    feat_remov = feat[feat > feat_cutoff].index
    x = x.drop(feat_remov, axis=1)

    # fill nas with mean
    x = x.fillna(x.mean())
    return x


def import_dataset_hospital(datset, outcome, hospital, na_cutoff, shuffle=True):
    # get source and target data
    x_source, y_source, x_test, y_test, feats = get_dataset_hospital(
        datset, outcome, hospital, na_cutoff
    )

    # get train, validation and test set
    x_train, x_val, y_train, y_val = train_test_split(
        x_source, y_source, train_size=0.67
    )

    # shuffle train and validation
    if shuffle:
        (x_train, y_train), (x_val, y_val) = random_shuffle_and_split(
            x_train, y_train, x_val, y_val, len(x_train)
        )

    orig_dims = x_train.shape[1:]

    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), feats, orig_dims


def random_shuffle_and_split(x_train, y_train, x_test, y_test, split_index):
    x = np.append(x_train, x_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    x, y = unison_shuffled_copies(x, y)

    x_train = x[:split_index, :]
    x_test = x[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return (x_train, y_train), (x_test, y_test)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def run_shift_experiment(
    shift,
    outcome,
    hospital,
    path,
    dr_technique,
    md_test,
    samples,
    dataset,
    sign_level,
    na_cutoff,
    random_runs=5,
    calc_acc=True,
):
    # Stores p-values for all experiments of a shift class.
    samples_rands = np.ones((len(samples), random_runs)) * (-1)

    print("Shift %s" % shift)
    shift_path = path + shift + "/"

    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs - 1):
        rand_run = int(rand_run)
        rand_run_path = shift_path + str(rand_run) + "/"
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        np.random.seed(rand_run)

        (
            (X_tr, y_tr),
            (X_val, y_val),
            (X_t, y_t),
            feats,
            orig_dims,
        ) = import_dataset_hospital(shift, outcome, hospital, na_cutoff, shuffle=True)

        # Run shift experiements across various sample sizes
        for si, sample in enumerate(samples):

            red_model = None
            print("Random Run %s : Sample %s" % (rand_run, sample))

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, red_model, sample, dataset
            )
            (
                p_val,
                dist,
                red_dim,
                red_model,
                val_acc,
                te_acc,
            ) = shift_detector.detect_data_shift(
                X_tr, y_tr, X_val, y_val, X_t[:sample, :], y_t[:sample], orig_dims
            )

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            print("Shift p-vals: ", p_val)

            samples_rands[si, rand_run] = p_val

    mean_p_vals = np.mean(samples_rands, axis=1)
    std_p_vals = np.std(samples_rands, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    return (mean_p_vals, std_p_vals)


def run_synthetic_shift_experiment(
    shift,
    outcome,
    hospital,
    path,
    dr_technique,
    md_test,
    samples,
    dataset,
    sign_level,
    na_cutoff,
    random_runs=5,
    calc_acc=True,
):
    # Stores p-values for all experiments of a shift class.
    samples_rands = np.ones((len(samples), random_runs)) * (-1)

    print("Shift %s" % shift)
    shift_path = path + shift + "/"
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs - 1):
        rand_run = int(rand_run)
        rand_run_path = shift_path + str(rand_run) + "/"
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        np.random.seed(rand_run)

        (
            (X_tr, y_tr),
            (X_val, y_val),
            (X_t, y_t),
            feats,
            orig_dims,
        ) = import_dataset_hospital(
            "baseline", outcome, hospital, na_cutoff, shuffle=True
        )
        X_t_1, y_t_1 = X_t.copy(), y_t.copy()
        (X_t_1, y_t_1) = apply_shift(X_tr, y_tr, X_t_1, y_t_1, shift)

        for si, sample in enumerate(samples):

            red_model = None
            print("Random Run %s : Sample %s" % (rand_run, sample))

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, red_model, sample, dataset
            )
            (
                p_val,
                dist,
                red_dim,
                red_model,
                val_acc,
                te_acc,
            ) = shift_detector.detect_data_shift(
                X_tr, y_tr, X_val, y_val, X_t_1[:sample, :], y_t_1[:sample], orig_dims
            )

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            print("Shift p-vals: ", p_val)

            samples_rands[si, rand_run] = p_val

    mean_p_vals = np.mean(samples_rands, axis=1)
    std_p_vals = np.std(samples_rands, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    return (mean_p_vals, std_p_vals)
