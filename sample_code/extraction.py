
import pandas as pd
import numpy as np
import json
import os
import time
import sqlalchemy
import pandas.io.sql as psql
import config as conf

def extract(config):
    print('postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}')
    engine = sqlalchemy.create_engine(f'postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}')

    # extract basic demographics and length of stay information from ip_administrative
    query = """select distinct
        i.patient_id_hashed as patient_id,
        i.genc_id,
        i.hospital_id,
        CASE when i.gender = 'F' THEN 1 ELSE 0 END AS sex,
        i.age,
        CASE when i.discharge_disposition = 7 THEN 1 ELSE 0 END AS mort_hosp,
	    i.discharge_date_time, 
	    i.admit_date_time,
	    f.diagnosis_code as mr_diagnosis,
	    DATE_PART('year', i.admit_date_time) as year, 
        (extract(epoch from i.discharge_date_time)::FLOAT - extract(epoch from i.admit_date_time)::float)/(24*60*60) as los,
        CASE when NULLIF(REPLACE(REPLACE(i.readmission, 'Yes', '9'), 'No', '5'), '')::numeric::integer = 2 or  NULLIF(REPLACE(REPLACE(i.readmission, 'Yes', '9'), 'No', '5'), '')::numeric::integer = 4  THEN 1 ELSE 0 END AS readmission_7,
        CASE when NULLIF(REPLACE(REPLACE(i.readmission, 'Yes', '9'), 'No', '5'), '')::numeric::integer = 2 or  NULLIF(REPLACE(REPLACE(i.readmission, 'Yes', '9'), 'No', '5'), '')::numeric::integer = 3 or  NULLIF(REPLACE(REPLACE(i.readmission, 'Yes', '9'), 'No', '5'), '')::numeric::integer = 4  THEN 1 ELSE 0 END AS readmission_28,
        CASE when g.pal =1 THEN 1 ELSE 0 END AS palliative,
        e.los_er,
        e.admit_via_ambulance,
        e.triage_date_time as er_admit_date_time,
        e.left_er_date_time as er_discharge_date_time,
        e.triage_level
      FROM ip_administrative i,
          LEFT OUTER JOIN (SELECT d.genc_id, d.diagnosis_code
                  FROM diagnosis d
                  WHERE d.diagnosis_type='M' AND d.is_er_diagnosis='FALSE') f
                  ON i.genc_id=f.genc_id
          LEFT OUTER JOIN (SELECT d.genc_id, 1 as pal
                  FROM diagnosis d
                  WHERE d.diagnosis_code = 'Z515') g
                  ON i.genc_id=g.genc_id
          LEFT OUTER JOIN (SELECT e.genc_id, 
                                e.admit_via_ambulance,
                                e.disposition_date_time,
                                e.duration_er_stay_derived AS los_er,
                                e.left_er_date_time,
                                e.physician_initial_assessment_date_time,
                                e.triate_date_time as triage_date_time,
                                e.triage_level
                  FROM er_administrative e) e
                  ON i.genc_id=e.genc_id
      ORDER BY patient_id, genc_id
      LIMIT 10000"""

    data=pd.read_sql(query ,con=engine)
    print(data.head())

    return data

def binary_legth_of_stay(l):
    return 1 if l >= 7 else 0    #TODO: replace with parameter

def transform(data):
    # convert length of stay feature
    # 1 - more then 7 days, 0 - less
    data["stay_length"]=data["stay_length"].apply(binary_legth_of_stay)
    print(data.columns.tolist())
    return data