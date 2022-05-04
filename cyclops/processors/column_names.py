"""Column names to use across datasets, used for processing."""


ENCOUNTER_ID = "encounter_id"
HOSPITAL_ID = "hospital_id"
ADMIT_TIMESTAMP = "admit_timestamp"
DISCHARGE_TIMESTAMP = "discharge_timestamp"
DISCHARGE_DISPOSITION = "discharge_disposition"
READMISSION = "readmission"

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

LENGTH_OF_STAY_IN_ER = "length_of_stay_in_er"
MORTALITY_IN_HOSPITAL = "mortality_in_hospital"

EVENT_NAME = "event_name"
EVENT_VALUE = "event_value"
EVENT_VALUE_UNIT = "event_value_unit"
EVENT_TIMESTAMP = "event_timestamp"

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
]
