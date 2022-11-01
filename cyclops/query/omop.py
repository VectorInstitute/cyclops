"""Query API functions using the OMOP mapping."""


# Table names.
VISIT_OCCURRENCE = "visit_occurrence"
VISIT_DETAIL = "visit_detail"
PERSON = "person"
MEASUREMENT = "measurement"
CONCEPT = "concept"
OBSERVATION = "observation"
CARE_SITE = "care_site"

# Table map.
TABLE_MAP = {
    VISIT_OCCURRENCE: lambda db: db.ods.visit_occurrence,
    VISIT_DETAIL: lambda db: db.ods.visit_detail,
    PERSON: lambda db: db.ods.person,
    MEASUREMENT: lambda db: db.ods.measurement,
    OBSERVATION: lambda db: db.ods.observation,
    CONCEPT: lambda db: db.ods.concept,
    CARE_SITE: lambda db: db.ods.care_site,
}

# OMOP column names.
VISIT_OCCURRENCE_ID = "visit_occurrence_id"
PERSON_ID = "person_id"
VISIT_START_DATETIME = "visit_start_datetime"
VISIT_END_DATETIME = "visit_end_datetime"
VISIT_DETAIL_START_DATETIME = "visit_detail_start_datetime"
VISIT_DETAIL_END_DATETIME = "visit_detail_end_datetime"
VISIT_CONCEPT_ID = "visit_concept_id"
VISIT_TYPE_CONCEPT_ID = "visit_type_concept_id"
VISIT_DETAIL_CONCEPT_ID = "visit_detail_concept_id"
VISIT_DETAIL_TYPE_CONCEPT_ID = "visit_detail_type_concept_id"
CARE_SITE_ID = "care_site_id"
CONCEPT_NAME = "concept_name"
CONCEPT_ID = "concept_id"
CARE_SITE_SOURCE_VALUE = "care_site_source_value"
OBSERVATION_CONCEPT_ID = "observation_concept_id"
OBSERVATION_TYPE_CONCEPT_ID = "observation_type_concept_id"
OBSERVATION_DATETIME = "observation_datetime"
MEASUREMENT_CONCEPT_ID = "measurement_concept_id"
MEASUREMENT_TYPE_CONCEPT_ID = "measurement_type_concept_id"
MEASUREMENT_DATETIME = "measurement_datetime"
UNIT_CONCEPT_ID = "unit_concept_id"
VALUE_AS_CONCEPT_ID = "value_as_concept_id"

# Created columns.
VISIT_DETAIL_CONCEPT_NAME = "visit_detail_concept_name"
CARE_SITE_NAME = "care_site_name"
GENDER_CONCEPT_NAME = "gender_concept_name"
RACE_CONCEPT_NAME = "race_concept_name"
ETHNICITY_CONCEPT_NAME = "ethnicity_concept_name"

# Column map.
OMOP_COLUMN_MAP: dict = {}
