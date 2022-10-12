"""Mortality decompensation use-case constants."""

import numpy as np

from cyclops.processors.column_names import AGE, DIAGNOSIS_TRAJECTORY, HOSPITAL_ID, SEX
from cyclops.processors.constants import TARGETS
from cyclops.utils.file import join, process_dir_save_path

CONST_NAME = "mortality_decompensation"
USECASE_ROOT_DIR = join(
    "/mnt/nfs/project/delirium",
    "cyclops",
    "usecases",
    "gemini",
    CONST_NAME,
)
DATA_DIR = process_dir_save_path(join(USECASE_ROOT_DIR, "./data"))

OUTCOME_DEATH = "outcome_death"
OUTCOME_EDEMA = "outcome_edema"
SPLIT_FRACTIONS = [0.8, 0.1, 0.1]

ENCOUNTERS_FILE = join(DATA_DIR, "encounters.parquet")
AGGREGATED_FILE = join(DATA_DIR, "aggregated.parquet")
TAB_FEATURES_FILE = join(DATA_DIR, "tab_features.pkl")
TAB_VECTORIZED_FILE = join(DATA_DIR, "tab_vectorized.pkl")
TEMP_VECTORIZED_FILE = join(DATA_DIR, "temp_vectorized.pkl")

# Tabular
TAB_TARGETS = [OUTCOME_DEATH, OUTCOME_EDEMA]
TAB_FEATURES = [
    HOSPITAL_ID,
    AGE,
    SEX,
    DIAGNOSIS_TRAJECTORY,
    OUTCOME_DEATH,
    OUTCOME_EDEMA,
    "readmission",
    "from_nursing_home_mapped",
    "from_acute_care_institution_mapped",
    "los_derived",
    "prev_encounter_count",
    "triage_level",
] + TAB_TARGETS

# Temporal
TIMESTEP_SIZE = 24
WINDOW_DURATION = 144
PREDICT_OFFSET = 24 * 14

TOP_N_EVENTS = 150

OUTCOME_DEATH_TEMP = TARGETS + " - " + OUTCOME_DEATH
TEMP_TARGETS = [OUTCOME_DEATH_TEMP]

TARGET_TIMESTAMP = "deathtime"

QUERIED_DIR = process_dir_save_path(join(DATA_DIR, "0-queried"))
CLEANED_DIR = process_dir_save_path(join(DATA_DIR, "1-cleaned"))
AGGREGATED_DIR = process_dir_save_path(join(DATA_DIR, "2-agg"))
VECTORIZED_DIR = process_dir_save_path(join(DATA_DIR, "3-vec"))

# Saving final vectorized
FINAL_VECTORIZED = process_dir_save_path(join(DATA_DIR, "4-final"))
TAB_UNALIGNED = join(FINAL_VECTORIZED, "unaligned_")
TAB_VEC_COMB = join(FINAL_VECTORIZED, "aligned_")

# Querying constants
BEFORE_DATE = "2020-01-23"

SEXES = ["M", "F"]
