"""Common use-case constants."""

import numpy as np

from cyclops.processors.column_names import AGE, DIAGNOSIS_TRAJECTORY, HOSPITAL_ID, SEX
from cyclops.processors.constants import TARGETS
from cyclops.utils.file import join, process_dir_save_path


OUTCOME_DEATH = "outcome_death"
OUTCOME_EDEMA = "outcome_edema"


# Querying constants
IMAGING_DESCRIPTIONS = ["X-ray", "CT", "MRI", "Ultrasound", "Echo"]

BT_SUBSTRINGS = ["platelet", "albumin", "plasma"]

DERIVED_VARIABLES = ["ip_charlson_derived"]

# Imaging notes must include of all the substrings to be classified as pulmonary edema.
EDEMA_IMAGING_SUBSTRINGS = ["pulmonary", "edema"]
EDEMA_PHARMA_SUBSTRINGS = ["lasix", "furosemide"]

PRESCRIPTION_AFTER_IMAGING_DAYS = 1

IMAGING_KEYWORDS = {
    "head": [
        "head",
        "facial",
        "sinus",
        "tm joint",
        "parotid",
        "mandible",
        "willis",
        "brain",
        "cerebral",
        "skull",
        "fossa",
    ],
    "neck": ["neck", "carotid", "thyroid"],
    "chest": [
        "chest",
        "lung",
        "rib",
        "sternum",
        "cardiac",
        "trachea",
        "thora",
        "pulmonary",
        "clavicle",
        "esophageal",
        "heart",
    ],
    "abd": [
        "abd",
        "liver",
        "pancreas",
        "colon",
        "lumbar",
        "enterography",
        "urogram",
        "gastr",
        "renal",
    ],
    "pelvis": [
        "pelvis",
        "hip",
        "sacrum",
        "coccyx",
        "testicle",
        "vagina",
        "tv",
        "sacroiliac",
        "scrotum",
    ],
    "limb": [
        "limb",
        "leg",
        "arm",
        "elbow",
        "humerus",
        "ulna",
        "radius",
        "wrist",
        "knee",
        "tib",
        "fib",
        "toe",
        "femur",
        "patella",
        "ankle",
        "feet",
        "foot",
        "peripheral",
        "extremity",
        "hand",
        "fger",
        "finger",
        "thumb",
    ],
    "shoulder": ["shoulder", "scapula"],
    "whole_body": ["spine", "whole body"],
}

READMISSION_MAP = {
    "planned_from_acute": [1, "1", "Yes"],
    "unplanned_7_day_acute": [2, "2"],
    "unplanned_8_to_28_day_acute": [3, "3"],
    "unplanned_7_day_day_surg": [4, "4"],
    "new_to_acute": [5, "5"],
    "nota": [9, "9", "No"],
    "no_info": np.nan,
}

ADMIT_VIA_AMBULANCE_MAP = {
    "ground": ["G", "GROUND", "GROUND AMBULANCE", "Y"],
    "no_ambulance": ["N", "No Ambulance"],
    "air": ["A", "C", "COMBINED", "COMBINATION OF AMBULANCES - Includes Air Ambulance"],
    "no_info": np.nan,
}

TRIAGE_LEVEL_MAP = {
    "resuscitation": ["1", "L1", "RESUSCITATION"],
    "emergent": ["2", "L2", "EMERGENCY"],
    "urgent": ["3", "L3", "URGENT"],
    "semi-urgent": ["4", "L4", "SEMI-URGENT"],
    "non-urgent": ["5", "NON-URGENT"],
    "no_info": ["9", np.nan],
}
