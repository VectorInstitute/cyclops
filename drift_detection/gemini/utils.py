import random
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
from sklearn.preprocessing import StandardScaler

from .constants import *

sys.path.append("..")

from drift_detector.detector import Detector
from drift_detector.reductor import Reductor
from drift_detector.experimenter import Experimenter

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

MORTALITY = "mortality"
LOS = "los"
AGGREGATION_WINDOW = 144
AGGREGATION_BUCKET_SIZE = 24

def apply_predefined_shift(experimenter, X_s_orig, y_s_orig, X_te_orig, y_te_orig, shift, tol_var="gender"):

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
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 10.0, normalization=1.0, delta=1.0, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_gn_shift_1.0":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 1.0, normalization=1.0, delta=1.0, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "small_gn_shift_1.0":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 0.1, normalization=1.0, delta=1.0, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "large_gn_shift_0.5":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 10.0, normalization=1.0, delta=0.5, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_gn_shift_0.5":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 1.0, normalization=1.0, delta=0.5, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "small_gn_shift_0.5":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 0.1, normalization=1.0, delta=0.5, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "large_gn_shift_0.1":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 10.0, normalization=1.0, delta=0.1, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_gn_shift_0.1":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 1.0, normalization=1.0, delta=0.1, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "small_gn_shift_0.1":
        X_te_1, _ = experimenter.gaussian_noise(
            X_te_orig, 0.1, normalization=1.0, delta=0.1, clip=False
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "ko_shift_0.1":
        X_te_1, y_te_1 = self.knockout_shift(X_te_orig, y_te_orig, 0, 0.1)
    elif shift == "ko_shift_0.5":
        X_te_1, y_te_1 = experimenter.knockout_shift(X_te_orig, y_te_orig, 0, 0.5)
    elif shift == "ko_shift_1.0":
        X_te_1, y_te_1 = experimenter.knockout_shift(X_te_orig, y_te_orig, 0, 1.0)
    elif shift == "cp_shift_0.75":
        X_te_1, y_te_1 = experimenter.changepoint_shift(
            X_s_orig, y_s_orig, X_te_orig, y_te_orig, 0, n_shuffle=0.75, rank=True
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "cp_shift_0.25":
        X_te_1, y_te_1 = experimenter.changepoint_shift(
            X_s_orig, y_s_orig, X_te_orig, y_te_orig, 0, n_shuffle=0.25, rank=True
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.75_krc_rec":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.75,
            keep_rows_constant=True,
              repermute_each_column=True,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.5_krc_rec":
         X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.5,
            keep_rows_constant=True,
            repermute_each_column=True,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.25_krc_rec":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.25,
            keep_rows_constant=True,
            repermute_each_column=True,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.75_krc":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.75,
            keep_rows_constant=True,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.5_krc":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.5,
            keep_rows_constant=True,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.25_krc":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.25,
            keep_rows_constant=True,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.75":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.75,
            keep_rows_constant=False,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.5":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.5,
            keep_rows_constant=False,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "mfa_shift_0.25":
        X_te_1, y_te_1 = experimenter.multiway_feat_association_shift(
            X_te_orig,
            y_te_orig,
            n_shuffle=0.25,
            keep_rows_constant=False,
            repermute_each_column=False,
        )
        y_te_1 = y_te_orig.copy()
    elif shift == "large_bn_shift_1.0":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.5, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_bn_shift_1.0":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.1, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == "small_bn_shift_1.0":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.01, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == "large_bn_shift_0.5":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.5, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_bn_shift_0.5":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.1, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == "small_bn_shift_0.5":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.01, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == "large_bn_shift_0.1":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.5, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == "medium_bn_shift_0.1":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.1, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == "small_bn_shift_0.1":
        X_te_1, _ = experimenter.binary_noise(X_te_orig, 0.01, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == "tolerance":
        X_te_1, _ = experimenter.tolerance(X_s_orig, y_s_orig, X_te_orig, y_te_orig, tol_var)
        y_te_1 = y_te_orig.copy()
    else:
        raise ValueError("Not a pre-defined shift, specify custom parameters using appropriate function")
    return (X_te_1, y_te_1)

def get_merged_data(BASE_DATA_PATH):

    print("Load data from feature handler...")
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
        
    return merged_static_temporal, temporal, static

def get_aggregated_events(BASE_DATA_PATH):
    print("Load data from aggregated events...")
    ## Aggregated events
    aggregated_events = load_dataframe(os.path.join(BASE_DATA_PATH, "aggregated_events"))
    timestep_start_timestamps = load_dataframe(
        os.path.join(BASE_DATA_PATH, "aggmeta_start_ts")
    )
    aggregated_events.loc[aggregated_events["timestep"] == 6]["event_name"].value_counts()
    aggregated_events = aggregated_events.loc[aggregated_events["timestep"] != 6]
    return(aggregated_events)

def get_encounters(BASE_DATA_PATH):
    print("Load data from admin data...")
    encounters_data = pd.read_parquet(os.path.join(BASE_DATA_PATH, "admin_er.parquet"))
    encounters_data[LOS] = (
    encounters_data[DISCHARGE_TIMESTAMP] - encounters_data[ADMIT_TIMESTAMP]
    )
    encounters_data_atleast_los_24_hrs = encounters_data.loc[
        encounters_data[LOS] >= pd.to_timedelta(24, unit="h")
    ]
    return(encounters_data_atleast_los_24_hrs)
    
def get_gemini_data(BASE_DATA_PATH):  
    # Get aggregated events
    aggregated_events = get_aggregated_events(BASE_DATA_PATH)
    # Get merged static + temporal data
    merged_static_temporal, temporal, static = get_merged_data(BASE_DATA_PATH)
    # Get encounters > 24hr los
    encounters_data_atleast_los_24_hrs = get_encounters(BASE_DATA_PATH)
        # Get mortality events
    encounters_mortality = encounters_data_atleast_los_24_hrs.loc[
        encounters_data_atleast_los_24_hrs[MORTALITY] == True
    ]
    # Get non-mortality events
    encounters_not_mortality = encounters_data_atleast_los_24_hrs.loc[
        encounters_data_atleast_los_24_hrs[MORTALITY] == False
    ]
    num_encounters_not_mortality = len(encounters_mortality)
    encounters_not_mortality_subset = encounters_not_mortality[
        0:num_encounters_not_mortality
    ]  
    # Combine mortality + non-mortality events
    encounters_train_val_test = pd.concat(
        [encounters_mortality, encounters_not_mortality_subset], ignore_index=True
    )
    # Get events the result in mortality within 2 weeks
    timeframe = 14  # days
    encounters_mortality_within_risk_timeframe = encounters_mortality.loc[
        encounters_mortality[LOS]
        <= pd.to_timedelta(timeframe * 24 + AGGREGATION_WINDOW, unit="h")
    ]
    mortality_events = convert_to_events(
        encounters_mortality_within_risk_timeframe,
        event_name=f"death",
        event_category="general",
        timestamp_col=DISCHARGE_TIMESTAMP,
    )
    mortality_events = pd.merge(
        mortality_events, encounters_mortality, on=ENCOUNTER_ID, how="inner"
    )
    mortality_events = mortality_events[
        [
            ENCOUNTER_ID,
            EVENT_NAME,
            EVENT_TIMESTAMP,
            ADMIT_TIMESTAMP,
            EVENT_VALUE,
            EVENT_CATEGORY,
        ]
    ]
    mortality_events[EVENT_VALUE] = 1
    # Get mortality labels
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
    timestep_end_timestamps = load_dataframe(os.path.join(BASE_DATA_PATH, "aggmeta_end_ts"))
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
    
    X = merged_static_temporal[
        np.in1d(temporal.index.get_level_values(0), static.index.get_level_values(0))
    ]

    return encounters_train_val_test, X, mortality_risk_targets

def get_dataset_hospital(admin_data, x, y, dataset, outcome, hospitals,train_frac=0.8):
    
    # filter hospital
    admin_data = admin_data.loc[
            admin_data['hospital_id'].isin(hospitals)
    ]
    encounter_ids = list(x.index.get_level_values(0).unique())
    x = x[np.in1d(x.index.get_level_values(0),admin_data[ENCOUNTER_ID])]
    
    # get source and target data
    x_s = None
    y_s = None
    x_t = None
    y_t = None

    # get experimental dataset
    if dataset == "covid":

        ids_source = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.date > datetime.date(2019, 1, 1)) 
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 2, 1)),
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            ((admin_data["admit_timestamp"].dt.date > datetime.date(2020, 3, 1)) 
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 8, 1)),
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "pre-covid":
        dataset_ids = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2019, 1, 1)) 
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2020, 2, 1)),
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac*len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_winter":
        ids_source = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.month.isin([11, 12, 1, 2]))
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.month.isin([6, 7, 8, 9]))
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_summer":
        ids_source = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.month.isin([6, 7, 8, 9]))
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.month.isin([11, 12, 1, 2]))
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_summer_baseline":
        dataset_ids = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.month.isin([6, 7, 8, 9]))
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        x_spl = np.array_split(x, 2)
        num_train = int(train_frac*len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "seasonal_winter_baseline":
        dataset_ids = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.month.isin([11, 12, 1, 2]))
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac*len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_academic":
        ids_source = admin_data.loc[
            (
                (admin_data["hospital_id"].isin(["SMH", "MSH", "UHNTG", "UHNTW","PMH"]))
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            (
                (admin_data["hospital_id"].isin(["THPC", "THPM"]))
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_academic_baseline":
        dataset_ids = admin_data.loc[
            (
                (admin_data["hospital_id"].isin(["SMH", "MSH", "UHNTG", "UHNTW","PMH"]))
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac*len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_community":
        ids_source = admin_data.loc[
            (
                (admin_data["hospital_id"].isin(["THPC", "THPM"]))
            ),
            "encounter_id",
        ]
        ids_target = admin_data.loc[
            (
                (admin_data["hospital_id"].isin(["SMH", "MSH", "UHNTG", "UHNTW","PMH"]))
            ),
            "encounter_id",
        ]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

    elif dataset == "hosp_type_community_baseline":
        dataset_ids = admin_data.loc[
            (
                (admin_data["hospital_id"].isin(["THPC", "THPM"]))
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac*len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]
        
    elif dataset == "simulated_deployment":
        dataset_ids = admin_data.loc[
            (
                (admin_data["admit_timestamp"].dt.date > datetime.date(2015, 1, 1)) 
                & (admin_data["admit_timestamp"].dt.date < datetime.date(2019, 1, 1)),
            ),
            "encounter_id",
        ]
        x = x.loc[x.index.get_level_values(0).isin(dataset_ids)]
        num_train = int(train_frac*len(dataset_ids))
        ids_source = dataset_ids[0:num_train]
        ids_target = dataset_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]

        
    elif dataset == "random":
        num_train = int(train_frac*len(encounter_ids))
        ids_source = encounter_ids[0:num_train]
        ids_target = encounter_ids[num_train:]
        x_s = x.loc[x.index.get_level_values(0).isin(ids_source)]
        x_t = x.loc[x.index.get_level_values(0).isin(ids_target)]
        
    y_s = y[np.in1d(encounter_ids, x_s.index.get_level_values(0).unique())]
    y_t = y[np.in1d(encounter_ids, x_t.index.get_level_values(0).unique())]
    
    assert len(x_s.index.get_level_values(0).unique()) == len(y_s)
    assert len(x_t.index.get_level_values(0).unique()) == len(y_t)
    
    return (x_s, y_s, x_t, y_t, x_s.columns, admin_data)


def import_dataset_hospital(admin_data, x, y, dataset, outcome, hospital, seed=1, shuffle=True, train_frac=0.8):
    # get source and target data
    x_source, y_source, x_test, y_test, feats, admin_data = get_dataset_hospital(
        admin_data, x, y, dataset,  outcome, hospital
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
        y_source[np.in1d(encounter_ids, ids)]
        for ids in [train_ids, val_ids]
    ]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), feats, admin_data

def get_label(admin_data, X, label):
    admin_data = admin_data.drop_duplicates('encounter_id')
    X_admin_data = admin_data[admin_data['encounter_id'].isin(X.index.get_level_values(0))]
    X_admin_data = X_admin_data.set_index('encounter_id').reindex(list(X.index.get_level_values(0).unique())).reset_index()
    y = X_admin_data["mortality"].astype(int)
    return y
    
def get_mortality_all(X_tr, X_val, X_t, admin_data):           
    y_tr = get_label(admin_data, X_tr, "mortality")
    y_val = get_label(admin_data, X_val, "mortality")
    y_t = get_label(admin_data, X_t, "mortality")
    return y_tr, y_val, y_t

def get_timeseries_mean(X_tr, X_val, X_t):
    X_tr_normalized = X_tr.groupby(level=[0]).mean()
    X_val_normalized = X_val.groupby(level=[0]).mean()
    X_t_normalized = X_t.groupby(level=[0]).mean()    
    return X_tr_normalized, X_val_normalized, X_t_normalized

def get_timeseries_first(X_tr, X_val, X_t):
    X_tr_normalized = X_tr.groupby(level=[0]).first()
    X_val_normalized = X_val.groupby(level=[0]).first()
    X_t_normalized = X_t.groupby(level=[0]).first()
    return X_tr_normalized, X_val_normalized, X_t_normalized

def get_timeseries_last(X_tr, X_val, X_t):
    X_tr_normalized = X_tr.groupby(level=[0]).last()
    X_val_normalized = X_val.groupby(level=[0]).last()
    X_t_normalized = X_t.groupby(level=[0]).last()
    return X_tr_normalized, X_val_normalized, X_t_normalized
  
def get_numerical_cols(path):
    feature_handler = FeatureHandler()
    feature_handler.load(path, "features")
    numerical_cols = feature_handler.get_numerical_feature_names()["temporal"]
    numerical_cols += ["age"]
    return numerical_cols
    
def scale_data(numerical_cols, X_tr_normalized, X_val_normalized, X_t_normalized):
    for col in numerical_cols:
        scaler = StandardScaler().fit(X_tr_normalized[col].values.reshape(-1, 1))
        X_tr_normalized[col] = pd.Series(
            np.squeeze(scaler.transform(X_tr_normalized[col].values.reshape(-1, 1))),
            index=X_tr_normalized[col].index,
        )
        X_val_normalized[col] = pd.Series(
            np.squeeze(scaler.transform(X_val_normalized[col].values.reshape(-1, 1))),
            index=X_val_normalized[col].index,
        )
        X_t_normalized[col] = pd.Series(
            np.squeeze(scaler.transform(X_t_normalized[col].values.reshape(-1, 1))),
            index=X_t_normalized[col].index,
        )
    return X_tr_normalized, X_val_normalized, X_t_normalized
  
def flatten(X_tr_normalized, X_val_normalized, X_t_normalized):
    X_tr_flattened = X_tr_normalized.unstack(1).dropna().to_numpy()
    X_val_flattened = X_val_normalized.unstack(1).dropna().to_numpy()
    X_t_flattened = X_t_normalized.unstack(1).dropna().to_numpy()
    return X_tr_flattened, X_val_flattened, X_t_flattened
 
def normalize_data(aggregation_type, admin_data, timesteps, X_tr, y_tr, X_val, y_val, X_t, y_t):
    if aggregation_type == "mean":
        X_tr_normalized, X_val_normalized, X_t_normalized = get_timeseries_mean(X_tr, X_val, X_t)       
        y_tr, y_val, y_t = get_mortality_all(X_tr, X_val, X_t, admin_data)
            
    elif aggregation_type == "first":
        X_tr_normalized, X_val_normalized, X_t_normalized = get_timeseries_first(X_tr, X_val, X_t)
        y_tr = y_tr[:,0]
        y_val = y_val[:,0]
        y_t = y_t[:,0]
            
    elif aggregation_type == "last":
        X_tr_normalized, X_val_normalized, X_t_normalized = get_timeseries_last(X_tr, X_val, X_t)   
        y_tr = y_tr[:,(timesteps-1)]
        y_val = y_val[:,(timesteps-1)]
        y_t = y_t[:,(timesteps-1)]
        
    elif aggregation_type == "time_flatten":
        X_tr_normalized = X_tr.copy()
        X_val_normalized = X_val.copy()
        X_t_normalized = X_t.copy()
        y_tr, y_val, y_t = get_mortality_all(X_tr, X_val, X_t, admin_data)
                        
    elif aggregation_type == "time":
        X_tr_normalized = X_tr.copy()
        X_val_normalized = X_val.copy()
        X_t_normalized = X_t.copy()    
    else:
        raise ValueError("Incorrect Aggregation Type")
    return (X_tr_normalized, y_tr),(X_val_normalized, y_val), (X_t_normalized, y_t)

def reshape_inputs(inputs, num_timesteps):
    inputs = inputs.unstack()
    num_encounters = inputs.shape[0]
    inputs = inputs.values.reshape((num_encounters, num_timesteps, -1))
    return inputs

def process_data(aggregation_type, timesteps, X_tr_normalized, X_val_normalized, X_t_normalized):
    if aggregation_type == "time_flatten":                    
        X_tr_input, X_val_input, X_t_input = flatten(X_tr_normalized, X_val_normalized, X_t_normalized)
    elif aggregation_type == "time":      
        X_tr_input = reshape_inputs(X_tr_normalized, timesteps)
        X_val_input = reshape_inputs(X_val_normalized, timesteps)
        X_t_input = reshape_inputs(X_t_normalized, timesteps)
    else:
        X_tr_input = X_tr_normalized.dropna().to_numpy()
        X_val_input = X_val_normalized.dropna().to_numpy()
        X_t_input = X_t_normalized.dropna().to_numpy()
        
    return X_tr_input, X_val_input, X_t_input

def run_shift_experiment(
    shift,
    admin_data,
    x, 
    y,
    outcome,
    hospital,
    path,
    aggregation_type,
    scale,
    dr_technique,
    model_path,
    md_test,
    context_type,
    representation,
    samples,
    dataset,
    sign_level,
    timesteps,
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
    
    numerical_cols = get_numerical_cols(path)
        
    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs):
        rand_run = int(rand_run)
        rand_run_path = shift_path + str(rand_run) + "/"
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        np.random.seed(rand_run)
        
        # Query data
        (X_tr, y_tr), (X_val, y_val), (X_t, y_t), feats, admin_data = import_dataset_hospital(admin_data, x, y, shift, outcome, hospital, rand_run, shuffle=True)
        
        # Normalize data
        (X_tr_normalized, y_tr),(X_val_normalized, y_val), (X_t_normalized, y_t) = normalize_data(aggregation_type, admin_data, timesteps, X_tr, y_tr, X_val, y_val, X_t, y_t)
        # Scale data
        if scale:
            X_tr_normalized, X_val_normalized, X_t_normalized = scale_data(numerical_cols, X_tr_normalized, X_val_normalized, X_t_normalized)

        # Process data
        X_tr_final, X_val_final, X_t_final = process_data(aggregation_type, timesteps, X_tr_normalized, X_val_normalized, X_t_normalized)
         
        # Run shift experiments across various sample sizes
        for si, sample in enumerate(samples):

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            shift_reductor = ShiftReductor(
            X_tr_final, y_tr, dr_technique, dataset, var_ret=0.8, model_path=model_path,
            )
            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, shift_reductor, sample, dataset, feats, model_path, context_type, representation,
            )
            
            try:
                (
                    p_val,
                    dist,
                    val_acc,
                    te_acc,
                ) = shift_detector.detect_data_shift(
                    X_tr_final, X_val_final[:1000,:], X_t_final[:sample,:]
                )
                
                print("Shift %s | Random Run %s | Sample %s | P-Value %s" % (shift, rand_run, sample, p_val))
                
                val_accs[rand_run, si] = val_acc
                te_accs[rand_run, si] = te_acc

                samples_rands_pval[si, rand_run] = p_val
                samples_rands_dist[si, rand_run] = dist
            
            except ValueError as e:
                print ('Error Type: ', type (e))
            
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
    admin_data,
    x, 
    y,
    outcome,
    hospital,
    path,
    aggregation_type,
    scale,
    dr_technique,
    model_path,
    md_test,
    context_type,
    representation,
    samples,
    dataset,
    sign_level,
    timesteps,
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

    numerical_cols = get_numerical_cols(path)
        
    # Average over a few random runs to quantify robustness.
    for rand_run in range(0, random_runs):
        rand_run = int(rand_run)
        rand_run_path = shift_path + str(rand_run) + "/"
        if not os.path.exists(rand_run_path):
            os.makedirs(rand_run_path)

        np.random.seed(rand_run)
        
        # Query data
        (X_tr, y_tr), (X_val, y_val), (X_t, y_t), feats, admin_data = import_dataset_hospital(admin_data, x, y, "random", outcome, hospital, rand_run, shuffle=True)
        # Normalize data
        (X_tr_normalized, y_tr),(X_val_normalized, y_val), (X_t_normalized, y_t) = normalize_data(aggregation_type, admin_data, timesteps, X_tr, y_tr, X_val, y_val, X_t, y_t)
        # Scale data
        if scale:
            X_tr_normalized, X_val_normalized, X_t_normalized = scale_data(numerical_cols, X_tr_normalized, X_val_normalized, X_t_normalized)

        # Process data
        X_tr_final, X_val_final, X_t_final = process_data(aggregation_type, timesteps, X_tr_normalized, X_val_normalized, X_t_normalized)
        
        X_t_1, y_t_1 = X_tr_final.copy(), y_t.copy()
        (X_t_1, y_t_1) = apply_shift(X_tr_final, y_tr, X_t_1, y_t_1, shift)
        
        # Run shift experiments across various sample sizes
        for si, sample in enumerate(samples):

            sample_path = rand_run_path + str(sample) + "/"

            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            shift_reductor = ShiftReductor(
            X_tr_final, y_tr, dr_technique, dataset, var_ret=0.8, model_path=model_path,
            )
            # Detect shift.
            shift_detector = ShiftDetector(
                dr_technique, md_test, sign_level, shift_reductor, sample, dataset, feats, model_path, context_type, representation, 
            )
            
            (
                p_val,
                dist,
                val_acc,
                te_acc,
            ) = shift_detector.detect_data_shift(
                X_tr_final, X_val_final[:1000,:], X_t_1[:sample,:],
            )
                
            print("Shift %s | Random Run %s | Sample %s | P-Value %s" % (shift, rand_run, sample, p_val))
            
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
    
