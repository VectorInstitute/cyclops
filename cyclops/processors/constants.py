"""Constants."""

# pylint: disable=line-too-long

# Generic.
YEAR = "year"
MONTH = "month"

# Feature handler.
FEATURES = "features"
TARGETS = "targets"
GROUP = "group"
BY = "by"

NAN_SUBSTITUTION_VALUE = "nan"

# Feature type.
NUMERIC = "numeric"
BINARY = "binary"
STRING = "string"
ORDINAL = "ordinal"
CATEGORICAL_INDICATOR = "categorical_indicator"

FEATURE_TYPES = [
    NUMERIC,
    BINARY,
    STRING,
    ORDINAL,
]

FEATURE_TYPE_ATTR = "type_"
FEATURE_TARGET_ATTR = "target"
FEATURE_INDICATOR_ATTR = "indicator_of"
FEATURE_MAPPING_ATTR = "mapping"

FEATURE_META_ATTRS = [
    FEATURE_TYPE_ATTR,
    FEATURE_TARGET_ATTR,
    FEATURE_INDICATOR_ATTR,
    FEATURE_MAPPING_ATTR,
]

FEATURE_META_ATTR_DEFAULTS = {
    FEATURE_TARGET_ATTR: False,
    FEATURE_INDICATOR_ATTR: None,
    FEATURE_MAPPING_ATTR: None,
}


MISSING_CATEGORY = "null_category"

# Feature normalization.
STANDARD = "standard"
MIN_MAX = "min-max"

# Aggregation / Imputation.
MEAN = "mean"
MEDIAN = "median"
MODE = "mode"
IGNORE = "ignore"
DROP = "drop"
FFILL = "ffill"
BFILL = "bffill"
FFILL_BFILL = "ffill_bfill"
LINEAR_INTERP = "linear_interp"

INTRA = "intra"
INTER = "inter"
EXTRA = "extra"

FIRST = "first"
LAST = "last"
ALL = "all"

# GEMINI Admin.
THPM = "THPM"
SBK = "SBK"
UHNTG = "UHNTG"
SMH = "SMH"
UHNTW = "UHNTW"
THPC = "THPC"
PMH = "PMH"
MSH = "MSH"
HOSPITAL_ID_MAP = {
    THPM: 0,
    SBK: 1,
    UHNTG: 2,
    SMH: 3,
    UHNTW: 4,
    THPC: 5,
    PMH: 6,
    MSH: 7,
}
MORTALITY_DISCHARGE_DISPOSITION_CODES = [7, 72, 73, 74]

# Diagnostic codes (ICD-10).
TRAJECTORIES = {
    "Certain infectious and parasitic diseases": ("A00", "B99"),
    "Neoplasms": ("C00", "D49"),
    "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism": (  # noqa: E501
        "D50",
        "D89",
    ),
    "Endocrine, nutritional and metabolic diseases": ("E00", "E89"),
    "Mental, Behavioral and Neurodevelopmental disorders": ("F01", "F99"),
    "Diseases of the nervous system": ("G00", "G99"),
    "Diseases of the eye and adnexa": ("H00", "H59"),
    "Diseases of the ear and mastoid process": ("H60", "H95"),
    "Diseases of the circulatory system": ("I00", "I99"),
    "Diseases of the respiratory system": ("J00", "J99"),
    "Diseases of the digestive system": ("K00", "K95"),
    "Diseases of the skin and subcutaneous tissue": ("L00", "L99"),
    "Diseases of the musculoskeletal system and connective tissue": ("M00", "M99"),
    "Diseases of the genitourinary system": ("N00", "N99"),
    "Pregnancy, childbirth and the puerperium": ("O00", "O99"),
    "Certain conditions originating in the perinatal period": ("P00", "P96"),
    "Congenital malformations, deformations and chromosomal abnormalities": (
        "Q00",
        "Q99",
    ),
    "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified": (  # noqa: E501
        "R00",
        "R99",
    ),
    "Injury, poisoning and certain other consequences of external causes": (
        "S00",
        "T88",
    ),
    "External causes of morbidity": ("V00", "Y99"),
    "COVID19": ("U07", "U08"),
    "Factors influencing health status and contact with health services": (
        "Z00",
        "Z99",
    ),
}

# GEMINI labs.
DRUG_SCREEN = [
    "amitriptyline",
    "amphetamine",
    "barbiturates",
    "barbiturates_scn",
    "barbiturates_and_sedatives_blood",
    "benzodiazepine_scn",
    "benzodiazepines_screen",
    "cannabinoids",
    "clozapine",
    "cocaine",
    "cocaine_metabolite",
    "codeine",
    "cocaine_metabolite",
    "codeine_metabolite_urine",
    "desipramine",
    "dextromethorphan",
    "dim_per_dip_metabolite",
    "dimen_per_diphenhydramine",
    "doxepin",
    "ephedrine_per_pseudo",
    "fluoxetine",
    "hydrocodone",
    "hydromorphone",
    "imipramine",
    "lidocaine",
    "mda_urine",
    "mdma_ecstacy",
    "methadone",
    "meperidine_urine",
    "methadone_metabolite_urine",
    "methamphetamine",
    "morphine",
    "morphine_metabolite_urine",
    "nortriptyline",
    "olanzapine_metabolite_u",
    "olanzapine_urine",
    "opiates_urine",
    "oxycodone",
    "oxycodone_cobas",
    "oxycodone_metabolite",
    "phenylpropanolamine",
    "propoxyphene",
    "sertraline",
    "trazodone",
    "trazodone_metabolite",
    "tricyclics_scn",
    "venlafaxine",
    "venlafaxine_metabolite",
]
EMPTY_STRING = ""
UNDERSCORE = "_"
NEGATIVE_RESULT_TERMS = [
    "neg",
    "not det",
    "no",
    "none seen",
    "arterial",
    "np",
    "non-reactive",
    "clear",
    "trace",
    "negative",
]
POSITIVE_RESULT_TERMS = [
    "pos",
    r"^\s*det",
    "yes",
    "venous",
    "present",
    "hazy",
    "slcloudy",
    "mild",
    "low reactive",
    "large",
    "turbid",
    "cloudy",
    "low reactive",
    "reactive",
    "positive",
]

# CARE UNIT CATEGORIES
IP = "IP"
ER = "ER"
ICU = "ICU"
SCU = "SCU"
