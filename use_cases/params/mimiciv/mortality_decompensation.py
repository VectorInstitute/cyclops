"""Mimiciv parameters for mortality decompensation data preprocessing."""

from cyclops.process.column_names import (
    ADMIT_TIMESTAMP,
    AGE,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    RESTRICT_TIMESTAMP,
    SEX,
    TIMESTEP,
)
from cyclops.process.constants import ALL, MEAN, STANDARD, TARGETS
from cyclops.process.impute import np_ffill_bfill
from cyclops.query.mimiciv import EVENTS
from cyclops.utils.file import join, process_dir_save_path

CONST_NAME = "mortality_decompensation"
OUTCOME_DEATH = "outcome_death"
DEATHTIME = "deathtime"

# PATHS
USECASE_ROOT_DIR = join(
    "/mnt/data",
    "cyclops",
    "use_cases",
    "mimiciv",
    CONST_NAME,
)
DATA_DIR = process_dir_save_path(join(USECASE_ROOT_DIR, "data"))

QUERIED_DIR = process_dir_save_path(join(DATA_DIR, "0-queried"))
CLEANED_DIR = process_dir_save_path(join(DATA_DIR, "1-cleaned"))
AGGREGATED_DIR = process_dir_save_path(join(DATA_DIR, "2-agg"))
VECTORIZED_DIR = process_dir_save_path(join(DATA_DIR, "3-vec"))
FINAL_VECTORIZED = process_dir_save_path(join(DATA_DIR, "4-final"))

TABULAR_FILE = join(DATA_DIR, "encounters.parquet")
TAB_FEATURES_FILE = join(DATA_DIR, "tab_features.pkl")
TAB_SLICE_FILE = join(DATA_DIR, "tab_slice.pkl")
TAB_AGGREGATED_FILE = join(DATA_DIR, "tab_aggregated.parquet")
TAB_VECTORIZED_FILE = join(DATA_DIR, "tab_vectorized.pkl")
TEMP_AGGREGATED_FILE = join(DATA_DIR, "temp_aggregated.parquet")
TEMP_VECTORIZED_FILE = join(DATA_DIR, "temp_vectorized.pkl")
COMB_VECTORIZED_FILE = join(DATA_DIR, "comb_vectorized.pkl")

ALIGNED_PATH = join(FINAL_VECTORIZED, "aligned_")
UNALIGNED_PATH = join(FINAL_VECTORIZED, "unaligned_")

# PARAMS
COMMON_FEATURE = ENCOUNTER_ID
SKIP_N = 0
SPLIT_FRACTIONS = [0.8, 0.1, 0.1]

TABULAR_FEATURES = {
    "primary_feature": ENCOUNTER_ID,
    "outcome": OUTCOME_DEATH,
    "targets": [OUTCOME_DEATH],
    "features": [AGE, SEX, "admission_type", "admission_location", OUTCOME_DEATH],
    "features_types": {},
}

TABULAR_NORM = {
    "normalize": True,
    "method": STANDARD,
}

TABULAR_SLICE = {
    "slice": False,
    "slice_map": {AGE: 80},
    "slice_query": "",
    "replace": True,
}

TABULAR_AGG = {
    "strategy": ALL,
    "index": ENCOUNTER_ID,
    "var_name": EVENT_NAME,
    "value_name": EVENT_VALUE,
}

TEMPORAL_PARAMS = {
    "query": EVENTS,
    "top_n_events": 150,
    "timestep_size": 24,
    "window_duration": 144,  # 24 * 6
    "predict_offset": 336,  # 24 * 14
}

TEMPORAL_NORM = {
    "normalize": True,
    "method": STANDARD,
}

TEMPORAL_FEATURES = {
    "primary_feature": EVENT_NAME,
    "features": [EVENT_VALUE],
    "groupby": [ENCOUNTER_ID, EVENT_NAME],
    "timestamp_col": EVENT_TIMESTAMP,
    "outcome": TARGETS + " - " + OUTCOME_DEATH,
    "targets": [TARGETS + " - " + OUTCOME_DEATH],
}


TIMESTAMPS = {
    "use_tabular": True,
    "columns": [ENCOUNTER_ID, ADMIT_TIMESTAMP, DISCHARGE_TIMESTAMP, DEATHTIME],
    "start_columns": [ENCOUNTER_ID, ADMIT_TIMESTAMP],
    "start_index": ENCOUNTER_ID,
    "rename": {ADMIT_TIMESTAMP: RESTRICT_TIMESTAMP},
}

TIMESTEPS = {"new_timestamp": "after_admit", "anchor": ADMIT_TIMESTAMP}

TEMPORAL_TARGETS = {
    "target_timestamp": DEATHTIME,
    "ref_timestamp": DISCHARGE_TIMESTAMP,
}

TEMPORAL_SLICE = {
    "slice": False,
    "slice_map": {},
    "replace": False,
}

TEMPORAL_AGG = {
    "aggfuncs": {EVENT_VALUE: MEAN},
    "timestamp_col": EVENT_TIMESTAMP,
    "time_by": ENCOUNTER_ID,
    "agg_by": [ENCOUNTER_ID, EVENT_NAME],
    "timestep_size": 24,
    "window_duration": 144,  # 24 * 6
}

TEMPORAL_IMPUTE = {
    "impute": True,
    "axis": TIMESTEP,
    "func": np_ffill_bfill,
}
