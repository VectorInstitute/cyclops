"""Query API functions using the OMOP mapping."""


# Table names.
VISIT_OCCURRENCE = "visit_occurrence"
VISIT_DETAIL = "visit_detail"
PERSON = "person"
MEASUREMENT = "measurement"
CONCEPT = "concept"
OBSERVATION = "observation"

# Table map.
TABLE_MAP = {
    VISIT_OCCURRENCE: lambda db: db.ods.visit_occurrence,
    VISIT_DETAIL: lambda db: db.ods.visit_detail,
    PERSON: lambda db: db.ods.person,
    MEASUREMENT: lambda db: db.ods.measurement,
    OBSERVATION: lambda db: db.ods.observation,
    CONCEPT: lambda db: db.ods.concept,
}

# Columns names.
VISIT_OCCURRENCE_ID = "visit_occurrence_id"
PERSON_ID = "person_id"
VISIT_START_DATETIME = "visit_start_datetime"
VISIT_END_DATETIME = "visit_end_datetime"
CARE_SITE_ID = "care_site_id"

# Column map.
OMOP_COLUMN_MAP = {}