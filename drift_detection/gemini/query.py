import os
import sys
import numpy as np
import pandas as pd
from .constants import *

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
