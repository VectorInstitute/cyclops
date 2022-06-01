"""Column names to use across datasets, used for processing."""


ENCOUNTER_ID = "encounter_id"
HOSPITAL_ID = "hospital_id"
ADMIT_TIMESTAMP = "admit_timestamp"
DISCHARGE_TIMESTAMP = "discharge_timestamp"
DISCHARGE_DISPOSITION = "discharge_disposition"
CARE_UNIT = "care_unit"
READMISSION = "readmission"
SCU_ADMIT_TIMESTAMP = "scu_admit_timestamp"
SCU_DISCHARGE_TIMESTAMP = "scu_discharge_timestamp"


AGE = "age"
SEX = "sex"
CITY = "city"
PROVINCE = "province"
COUNTRY = "country"
LANGUAGE = "language"
TOTAL_COST = "total_cost"
YEAR = "year"

DIAGNOSIS_CODE = "diagnosis_code"
DIAGNOSIS_TYPE = "diagnosis_type"
DIAGNOSIS_TRAJECTORIES = "diagnosis_trajectories"

ER_ADMIT_TIMESTAMP = "er_admit_timestamp"
ER_DISCHARGE_TIMESTAMP = "er_discharge_timestamp"
LENGTH_OF_STAY_IN_ER = "length_of_stay_in_er"
MORTALITY_IN_HOSPITAL = "mortality_in_hospital"

EVENT_NAME = "event_name"
EVENT_VALUE = "event_value"
EVENT_VALUE_UNIT = "event_value_unit"
EVENT_TIMESTAMP = "event_timestamp"

RESTRICT_TIMESTAMP = "restrict_timestamp"

REFERENCE_RANGE = "reference_range"
TIMESTEP = "timestep"

RECOGNISED_QUERY_COLUMNS = [
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
    LENGTH_OF_STAY_IN_ER,
    SEX,
    HOSPITAL_ID,
    DIAGNOSIS_CODE,
    DIAGNOSIS_TYPE,
    YEAR,
    AGE,
]
RECOGNISED_EVENT_COLUMNS = [
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    ENCOUNTER_ID,
    EVENT_NAME,
    EVENT_TIMESTAMP,
    EVENT_VALUE,
    EVENT_VALUE_UNIT,
]
