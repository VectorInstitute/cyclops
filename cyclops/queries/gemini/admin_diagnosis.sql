SELECT DISTINCT
    i.patient_id_hashed AS patient_id
    , i.genc_id
    , i.hospital_id
    , CASE WHEN i.gender = 'F' THEN
        1
    ELSE
        0
    END AS sex
    , i.age
    , CASE WHEN i.discharge_disposition = 7 THEN
        1
    ELSE
        0
    END AS mort_hosp
    , i.discharge_date_time
    , i.admit_date_time
    , f.diagnosis_code AS mr_diagnosis
    , DATE_PART('year' , i.admit_date_time) AS year
    , (extract(epoch FROM i.discharge_date_time)::FLOAT - extract(epoch FROM i.admit_date_time)::float) / (24 * 60 * 60) AS los
    , CASE WHEN NULLIF (REPLACE(REPLACE(i.readmission , 'Yes' , '9') , 'No' , '5') , '')::numeric::integer = 2
        OR NULLIF (REPLACE(REPLACE(i.readmission , 'Yes' , '9') , 'No' , '5') , '')::numeric::integer = 4 THEN
        1
    ELSE
        0
    END AS readmission_7
    , CASE WHEN NULLIF (REPLACE(REPLACE(i.readmission , 'Yes' , '9') , 'No' , '5') , '')::numeric::integer = 2
        OR NULLIF (REPLACE(REPLACE(i.readmission , 'Yes' , '9') , 'No' , '5') , '')::numeric::integer = 3
        OR NULLIF (REPLACE(REPLACE(i.readmission , 'Yes' , '9') , 'No' , '5') , '')::numeric::integer = 4 THEN
        1
    ELSE
        0
    END AS readmission_28
    , CASE WHEN g.pal = 1 THEN
        1
    ELSE
        0
    END AS palliative
    , e.los_er
    , e.admit_via_ambulance
    , e.triage_date_time AS er_admit_date_time
    , e.left_er_date_time AS er_discharge_date_time
    , e.triage_level
FROM
    ip_administrative i
    LEFT OUTER JOIN (
    SELECT
        d.genc_id
        , d.diagnosis_code
    FROM
        diagnosis d
    WHERE
        d.diagnosis_type = 'M'
        AND d.is_er_diagnosis = 'FALSE') f ON i.genc_id = f.genc_id
    LEFT OUTER JOIN (
    SELECT
        d.genc_id
        , 1 AS pal
    FROM
        diagnosis d
    WHERE
        d.diagnosis_code = 'Z515') g ON i.genc_id = g.genc_id
    LEFT OUTER JOIN (
    SELECT
        e.genc_id
        , e.admit_via_ambulance
        , e.disposition_date_time
        , e.duration_er_stay_derived AS los_er
        , e.left_er_date_time
        , e.physician_initial_assessment_date_time
        , e.triage_date_time AS triage_date_time
        , e.triage_level
    FROM
        er_administrative e) e ON i.genc_id = e.genc_id
ORDER BY
    patient_id
    , genc_id
