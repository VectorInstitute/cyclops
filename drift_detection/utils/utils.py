import datetime
import os
import pickle
import re
import sys
from collections import OrderedDict
from functools import reduce
import numpy as np
import pandas as pd
import sqlalchemy
from sklearn.model_selection import GridSearchCV, train_test_split
from sqlalchemy import desc, extract, func, select
from sqlalchemy.sql.expression import and_, or_

from .constants import *

sys.path.append("..")

from drift_detector.detector import ShiftDetector
from drift_detector.reductor import ShiftReductor
from experiments import apply_shift

sys.path.append("../..")

from cyclops.feature_handler import FeatureHandler
from cyclops.plotter import plot_timeline, set_bars_color, setup_plot
from cyclops.processor import run_data_pipeline
from cyclops.processors.aggregate import Aggregator
from cyclops.processors.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DIAGNOSIS_CODE,
    DISCHARGE_DISPOSITION,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    HOSPITAL_ID,
    LENGTH_OF_STAY_IN_ER,
    RESTRICT_TIMESTAMP,
    SEX,
    TIMESTEP,
    TRIAGE_LEVEL,
    WINDOW_START_TIMESTAMP,
)
from cyclops.processors.constants import SMH
from cyclops.processors.events import (
    combine_events,
    convert_to_events,
    normalize_events,
)
from cyclops.processors.impute import Imputer
from cyclops.processors.statics import compute_statics
from cyclops.processors.string_ops import replace_if_string_match, to_lower
from cyclops.processors.util import (
    create_indicator_variables,
    fill_missing_timesteps,
    gather_columns,
    pivot_aggregated_events_to_features,
)
from cyclops.query import gemini
from cyclops.utils.file import load_dataframe, save_dataframe


BASE_DATA_PATH = "/mnt/nfs/project/delirium/drift_exp/risk_of_mortality"

def get_data(hospital):

    hosp_label = "_".join(sorted(hospital, key=str.lower))
    file_name = os.path.join(BASE_DATA_PATH + "_" + hosp_label+ "_features.npy")

    if os.path.exists(file_name):
        x = np.load(file_name)
    else:
        ##### add code to get data ######
        print("Load data from feature handler")
        # Declare feature handler
        feature_handler = FeatureHandler()
        feature_handler.load(BASE_DATA_PATH, "features")
        
        # Get static and temporal data
        static = feature_handler.features["static"]
        temporal = feature_handler.features["temporal"]

         # Get types of columns
        numerical_cols = feature_handler.get_numerical_feature_names()["temporal"]
        cat_cols = feature_handler.get_categorical_feature_names()["temporal"]
        
        ## Impute numerical columns
        temporal[numerical_cols] = temporal[numerical_cols].ffill().bfill()

        # Check no more missingness!
        assert not temporal.isna().sum().sum() and not static.isna().sum().sum()
        
        # Combine static and temporal
        merged_static_temporal = temporal.combine_first(static)
        numerical_cols += ["age"]
        
        # Filter based on Hospital
        x = encounters[ENCOUNTER_ID].loc[
            encounters[HOSPITAL_ID].isin(hospital)
        ]
        
        np.save(filename, x)
        
    return x

def get_temporal_features(feature_handler):
    temporal_features = load_dataframe(os.path.join(BASE_DATA_PATH, "temporal_features"))
    feature_handler.add_features(temporal_features)
    feature_handler.drop_features(["death"])

    already_indicators = [
            "ct",
            "dialysis_mapped",
            "echo",
            "endoscopy_mapped",
            "interventional",
            "inv_mech_vent_mapped",
            "mri",
            "non-rbc",
            "other",
            "rbc",
            "surgery_mapped",
            "ultrasound",
            "unmapped_intervention",
            "x-ray",
    ]

    temporal_features = feature_handler.features["temporal"]
    indicators = create_indicator_variables(
        temporal_features[
            [col for col in temporal_features if col not in already_indicators]
        ]
    )
    feature_handler.add_features(indicators)
    feature_handler.features["temporal"].columns
        
def get_static_features(feature_handler):
    static_features = load_dataframe(os.path.join(BASE_DATA_PATH, "static_features"))
    feature_handler.add_features(
        static_features, reference_cols=[HOSPITAL_ID, ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP]
    )
    feature_handler.save(BASE_DATA_PATH, "features")

def create_mortality_labels(merged_static_temporal):
    num_timesteps = int(AGGREGATION_WINDOW / AGGREGATION_BUCKET_SIZE)
    encounter_ids = list(merged_static_temporal.index.get_level_values(0).unique())
    num_encounters = len(encounter_ids)

    # All zeroes.
    labels = np.zeros((num_encounters, num_timesteps))

    # Set mortality within timeframe encounters to all 1s.
    labels[
        [
            encounter_ids.index(enc_id)
            for enc_id in list(encounters_mortality_within_risk_timeframe[ENCOUNTER_ID])
        ]
    ] = 1

    # Get which timestep death occurs and set those and following timesteps label values to be -1.
    aggregated_mortality_events = aggregated_events.loc[
        aggregated_events[EVENT_NAME] == "death"
    ]
    for enc_id in list(aggregated_mortality_events[ENCOUNTER_ID]):
        timestep_death = aggregated_mortality_events.loc[
            aggregated_mortality_events[ENCOUNTER_ID] == enc_id
        ]["timestep"]
        labels[encounter_ids.index(enc_id)][int(timestep_death) + 1 :] = -1

    # Lookahead for each timestep, and see if death occurs in risk timeframe.
    for enc_id in list(encounters_mortality_within_risk_timeframe[ENCOUNTER_ID]):
        mortality_encounter = mortality_events.loc[mortality_events[ENCOUNTER_ID] == enc_id]
        ts_ends = timestep_end_timestamps.loc[enc_id]["timestep_end_timestamp"]
        mortality_ts = mortality_encounter["event_timestamp"]
        for ts_idx, ts_end in enumerate(ts_ends):
            if not (
                mortality_ts <= ts_end + pd.to_timedelta(timeframe * 24, unit="h")
            ).all():
                labels[encounter_ids.index(enc_id)][ts_idx] = 0


    mortality_risk_targets = labels
    return(mortality_risk_targets)

def get_dataset_hospital(datset, outcome, hospital):

    # get all data
    (admin_data, x) = get_data(hospital)

    # process features
    

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


def import_dataset_hospital(datset, outcome, hospital, shuffle=True):
    # get source and target data
    x_source, y_source, x_test, y_test, feats = get_dataset_hospital(
        datset, outcome, hospital
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

def run_pipeline(
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
    samples_rands_pval = np.ones((len(samples), random_runs)) * (-1)
    samples_rands_dist = np.ones((len(samples), random_runs)) * (-1)

    shift_path = path + shift + "/"

    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs):
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
        ) = import_dataset_hospital(shift, outcome, hospital, shuffle=True)
        
        # Run shift experiments across various sample sizes
        for si, sample in enumerate(samples):
            
            # print("Shift %s: Random Run %s : Sample %s" % (shift, rand_run, sample))

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            shift_reductor = ShiftReductor(
            X_tr, y_tr, dr_technique, orig_dims, dataset, dr_amount=None, var_ret=0.9, scale=False, scaler="standard", model=None
            )
            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, shift_reductor, sample, dataset
            )
            (
                p_val,
                dist,
                val_acc,
                te_acc,
            ) = shift_detector.detect_data_shift(
                X_tr, y_tr, X_val, y_val, X_t[:sample, :], y_t[:sample], orig_dims
            )

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            samples_rands_pval[si, rand_run] = p_val
            samples_rands_dist[si, rand_run] = dist
        
    mean_p_vals = np.mean(samples_rands_pval, axis=1)
    std_p_vals = np.std(samples_rands_pval, axis=1)
    
    mean_dist = np.mean(samples_rands_dist, axis=1)
    std_dist = np.std(samples_rands_dist, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    return (mean_p_vals, std_p_vals, mean_dist, std_dist)

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
    random_runs=5,
    calc_acc=True
):
    # Stores p-values for all experiments of a shift class.
    samples_rands_pval = np.ones((len(samples), random_runs)) * (-1)
    samples_rands_dist = np.ones((len(samples), random_runs)) * (-1)

    shift_path = path + shift + "/"

    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs):
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
        ) = import_dataset_hospital(shift, outcome, hospital, shuffle=True)
        
        # Run shift experiements across various sample sizes
        for si, sample in enumerate(samples):
            
            # print("Shift %s: Random Run %s : Sample %s" % (shift, rand_run, sample))

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            shift_reductor = ShiftReductor(
            X_tr, y_tr, dr_technique, orig_dims, dataset, dr_amount=None, var_ret=0.9, scale=False, scaler="standard", model=None
            )
            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, shift_reductor, sample, dataset
            )
            (
                p_val,
                dist,
                val_acc,
                te_acc,
            ) = shift_detector.detect_data_shift(
                X_tr, y_tr, X_val, y_val, X_t[:sample, :], y_t[:sample], orig_dims
            )

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            samples_rands_pval[si, rand_run] = p_val
            samples_rands_dist[si, rand_run] = dist
        
    mean_p_vals = np.mean(samples_rands_pval, axis=1)
    std_p_vals = np.std(samples_rands_pval, axis=1)
    
    mean_dist = np.mean(samples_rands_dist, axis=1)
    std_dist = np.std(samples_rands_dist, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    return (mean_p_vals, std_p_vals, mean_dist, std_dist)
    
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
    bucket_size=6, 
    window=24
):
    # Stores p-values for all experiments of a shift class.
    samples_rands_pval = np.ones((len(samples), random_runs)) * (-1)
    samples_rands_dist = np.ones((len(samples), random_runs)) * (-1)

    shift_path = path + shift + "/"
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)

    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs):
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
            "baseline", outcome, hospital, na_cutoff, bucket_size, window, shuffle=True
        )
        X_t_1, y_t_1 = X_t.copy(), y_t.copy()
        (X_t_1, y_t_1) = apply_shift(X_tr, y_tr, X_t_1, y_t_1, shift)

        for si, sample in enumerate(samples):

           # print("Shift %s: Random Run %s : Sample %s" % (shift, rand_run, sample))

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            shift_reductor = ShiftReductor(
            X_tr, y_tr, dr_technique, orig_dims, dataset, dr_amount=None, var_ret=0.9, scale=False, scaler="standard", model=None
            )
            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, shift_reductor, sample, dataset
            )
            (
                p_val,
                dist,
                val_acc,
                te_acc,
            ) = shift_detector.detect_data_shift(
                X_tr, y_tr, X_val, y_val, X_t_1[:sample, :], y_t_1[:sample], orig_dims
            )

            val_accs[rand_run, si] = val_acc
            te_accs[rand_run, si] = te_acc

            samples_rands_pval[si, rand_run] = p_val
            samples_rands_dist[si, rand_run] = dist
        
    mean_p_vals = np.mean(samples_rands_pval, axis=1)
    std_p_vals = np.std(samples_rands_pval, axis=1)
    
    mean_dist = np.mean(samples_rands_dist, axis=1)
    std_dist = np.std(samples_rands_dist, axis=1)

    mean_val_accs = np.mean(val_accs, axis=0)
    std_val_accs = np.std(val_accs, axis=0)

    mean_te_accs = np.mean(te_accs, axis=0)
    std_te_accs = np.std(te_accs, axis=0)

    return (mean_p_vals, std_p_vals, mean_dist, std_dist)
