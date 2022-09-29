"""WangLab use case constants."""

import os

from cyclops.processors.column_names import AGE, DIAGNOSIS_TRAJECTORY, HOSPITAL_ID, SEX
from cyclops.processors.constants import TARGETS
from cyclops.utils.file import join, process_dir_save_path

CONST_NAME = "mortality_decompensation"
USECASE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = process_dir_save_path(os.path.join(USECASE_ROOT_DIR, "./data"))

OUTCOME_DEATH = "outcome_death"
SPLIT_FRACTIONS = [0.8, 0.1, 0.1]

ENCOUNTERS_FILE = join(DATA_DIR, "encounters.parquet")
AGGREGATED_FILE = join(DATA_DIR, "aggregated.parquet")
TAB_FEATURES_FILE = join(DATA_DIR, "tab_features.pkl")
TAB_VECTORIZED_FILE = join(DATA_DIR, "tab_vectorized.pkl")
TEMP_VECTORIZED_FILE = join(DATA_DIR, "temp_vectorized.pkl")

# Tabular
TAB_TARGETS = [OUTCOME_DEATH]
TAB_FEATURES = [
    HOSPITAL_ID,
    AGE,
    SEX,
    DIAGNOSIS_TRAJECTORY,
    OUTCOME_DEATH,
] + TAB_TARGETS

# Temporal
TIMESTEP_SIZE = 24
WINDOW_DURATION = 144
PREDICT_OFFSET = 24 * 14

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
YEARS = list(range(2015, 2021))

# Column add-ons
MORTALITY = "mortality"
LOS = "los"
