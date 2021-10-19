
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
        LOWER(i.country) as country,
	    i.discharge_date_time, 
	    i.admit_date_time,
	    DATE_PART('year', i.admit_date_time) as year, 
	    EXTRACT (
          DAY
           FROM(i.discharge_date_time - i.admit_date_time)) 
           as stay_length,
        i.institution_from,
        i.institution_from_type,
        i.discharge_disposition::integer,
        i.institution_to,
        i.institution_to_type,
	i.province_territory_issuing_health_card_number as insurance
      FROM ip_administrative i
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