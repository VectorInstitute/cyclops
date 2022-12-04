"""Mortality decompensation use-case constants."""

from cyclops.process.column_names import AGE, DIAGNOSIS_TRAJECTORY, HOSPITAL_ID, SEX
from cyclops.process.constants import TARGETS
from cyclops.utils.file import join, process_dir_save_path

CONST_NAME = "mortality"
USECASE_ROOT_DIR = join(
    "/mnt/nfs/project/delirium",
    "drift_exp",
    "OCT-18-2022",
    "gemini",
    CONST_NAME,
)
DATA_DIR = process_dir_save_path(join(USECASE_ROOT_DIR, "./data"))

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
    "readmission",
    "from_nursing_home_mapped",
    "from_acute_care_institution_mapped",
    "prev_encounter_count",
    "triage_level",
    "admit_via_ambulance",
] + TAB_TARGETS
TAB_FEATURES_TYPES: dict = {}

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
BEFORE_DATE = "2020-08-1"

SEXES = ["M", "F"]
