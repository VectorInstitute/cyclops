"""Querying functions for GEMINI Use-case."""
import os
import sys

import numpy as np
import pandas as pd

# from cyclops.feature_handler import FeatureHandler
from cyclops.process.column_names import (
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_CATEGORY,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
)

# from cyclops.processors.events import convert_to_events
from cyclops.utils.file import load_dataframe

from .constants import AGGREGATION_BUCKET_SIZE, AGGREGATION_WINDOW, LOS, MORTALITY

sys.path.append("../..")


def get_merged_data(base_data_path):
    """Merge encounters and aggregated events.

    Parameters
    ----------
    base_data_path : str
        Path to the base data directory

    Returns
    -------
    pd.DataFrame
        Merged data
    temporal: pd.DataFrame
        Temporal data
    static: pd.DataFrame
        Static data

    """
    print("Load data from feature handler...")
    # Declare feature handler
    # No module named FeatureHandler in cyclops
    # feature_handler = FeatureHandler()
    feature_handler = callable()
    feature_handler.load(base_data_path, "features")

    # Get static and temporal data
    static = feature_handler.features["static"]
    temporal = feature_handler.features["temporal"]

    # Get types of columns
    numerical_cols = feature_handler.get_numerical_feature_names()["temporal"]
    # feature_handler.get_categorical_feature_names()["temporal"]

    # Impute numerical columns
    temporal[numerical_cols] = temporal[numerical_cols].ffill().bfill()

    # Check no more missingness!
    assert not temporal.isna().sum().sum() and not static.isna().sum().sum()

    # Combine static and temporal
    merged_static_temporal = temporal.combine_first(static)
    numerical_cols += ["age"]

    return merged_static_temporal, temporal, static


def get_aggregated_events(base_data_path):
    """Get aggregated events.

    Parameters
    ----------
    base_data_path : str
        Path to the base data directory

    Returns
    -------
    pd.DataFrame
        Aggregated events

    """
    print("Load data from aggregated events...")
    # Aggregated events
    aggregated_events = load_dataframe(
        os.path.join(base_data_path, "aggregated_events")
    )
    # not used
    # timestep_start_timestamps = load_dataframe(
    #     os.path.join(base_data_path, "aggmeta_start_ts")
    # )
    aggregated_events.loc[aggregated_events["timestep"] == 6][
        "event_name"
    ].value_counts()
    aggregated_events = aggregated_events.loc[aggregated_events["timestep"] != 6]
    return aggregated_events


def get_encounters(base_data_path):
    """Get encounters.

    Parameters
    ----------
    base_data_path : str
        Path to the base data directory

    Returns
    -------
    pd.DataFrame
        Encounters

    """
    print("Load data from admin data...")
    encounters_data = pd.read_parquet(os.path.join(base_data_path, "admin_er.parquet"))
    encounters_data[LOS] = (
        encounters_data[DISCHARGE_TIMESTAMP] - encounters_data[ADMIT_TIMESTAMP]
    )
    encounters_data_atleast_los_24_hrs = encounters_data.loc[
        encounters_data[LOS] >= pd.to_timedelta(24, unit="h")
    ]
    return encounters_data_atleast_los_24_hrs


def get_gemini_data(base_data_path):
    """Get GEMINI data.

    Parameters
    ----------
    base_data_path : str
        Path to the base data directory
    encounters_train_val_test, X, mortality_risk_targets
    Returns
    -------
    encounters_train_val_test : pd.DataFrame
        encounters metadata
    X : pd.DataFrame
        features
    mortality_risk_targets : pd.DataFrame
        mortality risk targets

    """
    # Get aggregated events
    aggregated_events = get_aggregated_events(base_data_path)
    # Get merged static + temporal data
    merged_static_temporal, temporal, static = get_merged_data(base_data_path)
    # Get encounters > 24hr los
    encounters_data_atleast_los_24_hrs = get_encounters(base_data_path)
    # Get mortality events
    encounters_mortality = encounters_data_atleast_los_24_hrs.loc[
        encounters_data_atleast_los_24_hrs[MORTALITY] is True
    ]
    # Get non-mortality events
    encounters_not_mortality = encounters_data_atleast_los_24_hrs.loc[
        encounters_data_atleast_los_24_hrs[MORTALITY] is False
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
    # No function named convert_to_events in cyclops.processors.events
    mortality_events = callable()
    # mortality_events = convert_to_events(
    #     encounters_mortality_within_risk_timeframe,
    #     event_name="death",
    #     event_category="general",
    #     timestamp_col=DISCHARGE_TIMESTAMP,
    # )
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
    # Get which timestep death occurs and set those and following timesteps
    # label values to be -1.
    aggregated_mortality_events = aggregated_events.loc[
        aggregated_events[EVENT_NAME] == "death"
    ]
    for enc_id in list(aggregated_mortality_events[ENCOUNTER_ID]):
        timestep_death = aggregated_mortality_events.loc[
            aggregated_mortality_events[ENCOUNTER_ID] == enc_id
        ]["timestep"]
        labels[encounter_ids.index(enc_id)][int(timestep_death) + 1 :] = -1
    timestep_end_timestamps = load_dataframe(
        os.path.join(base_data_path, "aggmeta_end_ts")
    )
    # Lookahead for each timestep, and see if death occurs in risk timeframe.
    for enc_id in list(encounters_mortality_within_risk_timeframe[ENCOUNTER_ID]):
        mortality_encounter = mortality_events.loc[
            mortality_events[ENCOUNTER_ID] == enc_id
        ]
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
