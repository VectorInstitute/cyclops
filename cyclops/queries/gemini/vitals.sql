SELECT
    i.patient_id_hashed AS patient_id
    , i.genc_id
    , i.hospital_id
    , v.measurement_name
    , v.measure_date_time
    , v.measurement_code
    , v.measurement_value
FROM
    (SELECT * FROM ip_administrative WHERE hospital_id = 'SMH' ORDER BY genc_id) i
    LEFT OUTER JOIN (
    SELECT
        v.genc_id
        , v.measurement_name
        , v.measure_date_time
        , v.measurement_code
        , v.measurement_value
    FROM
        vitals v) v ON i.genc_id = v.genc_id
ORDER BY
    patient_id
    , genc_id