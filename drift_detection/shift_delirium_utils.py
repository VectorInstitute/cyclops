import sys
from functools import reduce

sys.path.append("..")

import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import select, func, extract, desc
from sqlalchemy.sql.expression import and_

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataQualityTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataQualityProfileSection

import config
import cyclops
from cyclops.processors.column_names import (
    ENCOUNTER_ID,
    HOSPITAL_ID,
    ADMIT_TIMESTAMP,
    DISCHARGE_TIMESTAMP,
    AGE,
    SEX,
    TOTAL_COST,
    CITY,
    PROVINCE,
    COUNTRY,
    LANGUAGE,
    VITAL_MEASUREMENT_NAME,
    VITAL_MEASUREMENT_VALUE,
    VITAL_MEASUREMENT_TIMESTAMP,
    REFERENCE_RANGE,
)
from cyclops.processors.constants import EMPTY_STRING
from cyclops.processors.admin import AdminProcessor
from cyclops.processors.vitals import VitalsProcessor
from cyclops.processors.feature_handler import FeatureHandler
from cyclops.orm import Database

%load_ext autoreload
%load_ext nb_black

## load config, set up ORM
cfg = config.read_config("../configs/default/*.yaml")
db = Database(cfg)

def get_medication():
    # medication data query
    medication_name = func.lower(db.public.pharmacy.med_id_generic_name_raw)
    pharmacy_query = select(db.public.pharmacy.x,
                            medication_name.label(MEDICATION_NAME)).subquery()
    return(pharmacy_query)

def get_diagnosis():
    # diagnosis data query
    diagnosis_query = select(db.public.diagnosis.genc_id,
                         db.public.diagnosis.diagnosis_code,
                         db.public.diagnosis.diagnosis_type,
                         db.public.diagnosis.is_er_diagnosis).subquery()
    return(diagnosis_query)

def get_labs():
    # lab data query
    lab_test_name = func.lower(db.public.lab.lab_test_name_mapped)
    lab_query = select(db.public.lab.genc_id,
                   lab_test_name.label(LAB_TEST_NAME),
                   db.public.lab.result_value,
                   db.public.lab.result_unit,
                   db.public.lab.sample_collection_date_time).subquery()
    return(lab_query)

def get_data(admin=True,diagnosis=True,labs=True,medication=True):
    
    lab_query = get_labs()
    diagnosis_query = get_diagnosis()
    pharmacy_query = get_medication()
    
    if not labs and not medication:
        
        query = select(db.public.ip_administrative.patient_id_hashed.label(PATIENT_ID),
                   db.public.ip_administrative.genc_id,
                   db.public.ip_administrative.hospital_id,
                   db.public.ip_administrative.age,
                   db.public.ip_administrative.gender,
                   db.public.ip_administrative.admit_date_time,
                   db.public.ip_administrative.discharge_date_time,
                   db.public.ip_administrative.discharge_disposition,
                   db.public.ip_administrative.del_present,
                   db.public.ip_administrative.gemini_cohort,
                   diagnosis_query.c.diagnosis_code,
                   diagnosis_query.c.diagnosis_type,
                   diagnosis_query.c.is_er_diagnosis
                ).join(diagnosis_query,
                         db.public.ip_administrative.genc_id == diagnosis_query.c.genc_id
                        ).where(db.public.ip_administrative.gemini_cohort == True)
        
    elif labs and not medication:
        
        query = select(db.public.ip_administrative.patient_id_hashed.label(PATIENT_ID),
                   db.public.ip_administrative.genc_id,
                   db.public.ip_administrative.hospital_id,
                   db.public.ip_administrative.age,
                   db.public.ip_administrative.gender,
                   db.public.ip_administrative.admit_date_time,
                   db.public.ip_administrative.discharge_date_time,
                   db.public.ip_administrative.discharge_disposition,
                   db.public.ip_administrative.del_present,
                   db.public.ip_administrative.gemini_cohort,
                   diagnosis_query.c.diagnosis_code,
                   diagnosis_query.c.diagnosis_type,
                   diagnosis_query.c.is_er_diagnosis,
                   lab_query.c.lab_test_name,
                   lab_query.c.result_value,
                   lab_query.c.result_unit,
                   lab_query.c.sample_collection_date_time
                  ).join(lab_query, 
                         db.public.ip_administrative.genc_id == lab_query.c.genc_id).join(diagnosis_query,
                         db.public.ip_administrative.genc_id == diagnosis_query.c.genc_id
                        ).where(and_(lab_query.c.lab_test_name != '',
                                     db.public.ip_administrative.gemini_cohort == True))
     
    elif admin and diagnosis and labs and medication:
        
        query = select(db.public.ip_administrative.patient_id_hashed.label(PATIENT_ID),
                   db.public.ip_administrative.genc_id,
                   db.public.ip_administrative.hospital_id,
                   db.public.ip_administrative.age,
                   db.public.ip_administrative.gender,
                   db.public.ip_administrative.admit_date_time,
                   db.public.ip_administrative.discharge_date_time,
                   db.public.ip_administrative.discharge_disposition,
                   db.public.ip_administrative.del_present,
                   db.public.ip_administrative.gemini_cohort,
                   diagnosis_query.c.diagnosis_code,
                   diagnosis_query.c.diagnosis_type,
                   diagnosis_query.c.is_er_diagnosis,
                   lab_query.c.lab_test_name,
                   lab_query.c.result_value,
                   lab_query.c.result_unit,
                   lab_query.c.sample_collection_date_time,
                   pharmacy_query
                  ).join(lab_query, 
                         db.public.ip_administrative.genc_id == lab_query.c.genc_id).join(diagnosis_query,
                         db.public.ip_administrative.genc_id == diagnosis_query.c.genc_id
                        ).join(pharmacy_query,db.public.ip_administrative.genc_id == 
                               pharmacy_query.c.genc_id).where(and_(lab_query.c.lab_test_name != '',
                                     db.public.ip_administrative.gemini_cohort == True))
    # remove rows with no medication
    data = db.run_query(query)
    return(data)

def process_labs(data):
    return(data)

def process_admin(data):
    must_have_columns = [ENCOUNTER_ID, AGE, SEX]

    admin_processor = AdminProcessor(data, must_have_columns)
    admin_features = admin_processor.process()
    return(admin_features)

def process_vitals(data):
    must_have_columns = [
    ENCOUNTER_ID,
        ADMIT_TIMESTAMP,
        VITAL_MEASUREMENT_NAME,
        VITAL_MEASUREMENT_VALUE,
        VITAL_MEASUREMENT_TIMESTAMP,
        REFERENCE_RANGE,
    ]

    vitals_processor = VitalsProcessor(data, must_have_columns)
    vitals_features = vitals_processor.process()
    return(vitals_features)

def import_dataset(outcome, shuffle=True):
    ADMIN=True
    DIAGNOSIS=True
    LABS=True
    MEDICATION=False

    print('Loading data...')

    data = get_data(ADMIN, DIAGNOSIS,LABS,MEDICATION)

    #if baseline:
    #    print('baseline')
    #elif:
    #    print('other')
    
    
    data = data.loc[data['hospital_id'].isin([3])]
    data[features] = data[features].fillna(0)
    x = data[features].to_numpy()
    y = data[outcome].to_numpy()
    
    # Add 3-way split
    x_spl = np.array_split(x, 3)
    y_spl = np.array_split(y, 3)
    x_train = x_spl[0] ; x_val = x_spl[1] ; x_test = x_spl[2]
    y_train = y_spl[0] ; y_val = y_spl[1] ; y_test = x_spl[2]
    
    if shuffle:
        (x_train, y_train), (x_val, y_val) = random_shuffle_and_split(x_train, y_train, x_val, y_val, len(x_train))
        
    orig_dims = x_train.shape[1:]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), orig_dims
