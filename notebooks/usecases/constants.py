"""WangLab cardiac use case constants."""

BEFORE_DATE = "2020-01-23"

SEX = ["M", "F"]

DESCRIPTIONS = ["X-ray", "CT", "MRI", "Ultrasound", "Echo"]

BT_SUBSTRINGS = ["platelet", "albumin", "plasma"]

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