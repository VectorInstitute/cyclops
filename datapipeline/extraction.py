
import pandas as pd
import numpy as np
import json
import os
import time
import sqlalchemy
import pandas.io.sql as psql
import datapipeline.config as conf
import re

# constants
HOSPITAL_ID = {'THPM':0, 'SBK':1, 'UHNTG':2, 'SMH':3, 'UHNTW':4, 'THPC':5, 'PMH':6, 'MSH':7}
TRAJECTORIES = {
        'Certain infectious and parasitic diseases': ('A00', 'B99'),
        'Neoplasms': ('C00', 'D49'),
        'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': ('D50','D89'),
        'Endocrine, nutritional and metabolic diseases': ('E00', 'E89'),
        'Mental, Behavioral and Neurodevelopmental disorders': ('F01', 'F99'),
        'Diseases of the nervous system': ('G00', 'G99'),
        'Diseases of the eye and adnexa': ('H00', 'H59'),
        'Diseases of the ear and mastoid process': ('H60', 'H95'),
        'Diseases of the circulatory system': ('I00', 'I99'),
        'Diseases of the respiratory system': ('J00', 'J99'),
        'Diseases of the digestive system': ('K00', 'K95'),
        'Diseases of the skin and subcutaneous tissue': ('L00', 'L99'),
        'Diseases of the musculoskeletal system and connective tissue': ('M00', 'M99'),
        'Diseases of the genitourinary system': ('N00', 'N99'),
        'Pregnancy, childbirth and the puerperium': ('O00', 'O99'),
        'Certain conditions originating in the perinatal period': ('P00', 'P96'),
        'Congenital malformations, deformations and chromosomal abnormalities': ('Q00','Q99'),
        'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': ('R00', 'R99'),
        'Injury, poisoning and certain other consequences of external causes': ('S00', 'T88'),
        'External causes of morbidity': ('V00', 'Y99'),
        'COVID19': ('U07', 'U08'),
        'Factors influencing health status and contact with health services': ('Z00', 'Z99')
    }

def extract(config):
    print('postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}')
    engine = sqlalchemy.create_engine(f'postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}')

    pop_size = '' if config.pop_size == 0 else f'limit {config.pop_size}'
    filter = f"WHERE DATE_PART('year', i.admit_date_time) <= {int(config.filter_year)}" if config.filter_year else ''
    filter = f"WHERE i.admit_date_time  > '{config.filter_date_from}' AND i.admit_date_time <= '{config.filter_date_to}'" if config.filter_date_from else filter

    # extract basic demographics and length of stay information from ip_administrative
    query_full = f"""select distinct
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
      FROM ip_administrative i
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
                                e.triage_date_time as triage_date_time,
                                e.triage_level
                  FROM er_administrative e) e
                  ON i.genc_id=e.genc_id
      {filter}
      ORDER BY patient_id, genc_id
      {pop_size}"""

    data=pd.read_sql(query_full,con=engine)
    print(data.head())

    return data

def binary_legth_of_stay(l):
    return 1 if l >= 7 else 0    #TODO: replace with parameter

def insert_decimal(string, index=2):
    return string[:index] + '.' + string[index:]

def get_category(code, trajectories=TRAJECTORIES):
    """
    Usage:
    df['ICD10'].apply(get_category, args=(trajectories,))
    """
    if code is None:
        return np.nan
    try:
        code = str(code)
    except:
        return np.nan
    for item, value in trajectories.items():
        # check that code is greater than value_1
        if (re.sub('[^a-zA-Z]', '', code).upper() > value[0][0].upper()):
            # example, code is T and comparator is S
            pass
        elif (re.sub('[^a-zA-Z]', '', code).upper() == value[0][0].upper()) and (
                float(insert_decimal(re.sub('[^0-9]', '', code), index=2)) >= int(value[0][1:])):
            # example S21 > s00
            pass
        else:
            continue

        # check that code is less than value_2
        if (re.sub('[^a-zA-Z]', '', code).upper() < value[1][0].upper()):
            # example, code is S and comparator is T
            #             print(value[0], code, value[1])
            return "_".join(value)
        elif (re.sub('[^a-zA-Z]', '', code).upper() == value[1][0].upper()) and (
                int(float(insert_decimal(re.sub('[^0-9]', '', code), index=2))) <= int(value[1][1:])):
            # example S21 > s00
            #             print(value[0], code, value[1])
            return "_".join(value)
        else:
            continue
    raise Exception("Code cannot be converted: {}".format(code))

def transform_diagnosis(data):
    # apply the categorical ICD10 filter and one hot encode:
    data = pd.concat((data, pd.get_dummies(data.loc[:, 'mr_diagnosis'].apply(get_category, args=(TRAJECTORIES,)), dummy_na=True, columns=TRAJECTORIES.keys(), prefix='icd10')), axis=1)
    
    return data

# add a column to signal training/val or test
def split (data, config):
    #     Create the train and test folds: default test set is 2015. All patients in 2015 will be not be used for training 
    #     or validation. Default validation year is 2014. All patients in the validation year will not be used for training.
    #TODO: implement configurable train set - getting all except val/test in train set right now.

    #
    # set a new column for use_train_val
    data['train'] = 1
    data['test'] = 0
    data['val'] = 0
    if (config.split_column in ('year', 'hospital_id')):
        test_const = int(config.test)
        val_const = int (config.val)    
    else:
        test_const = config.test
        val_const = config.val                
    data.loc[data[config.split_column] == test_const, 'train'] = 0
    data.loc[data[config.split_column] == test_const, 'test'] = 1
    data.loc[data[config.split_column] == test_const, 'val'] = 0
    data.loc[data[config.split_column] == val_const, 'train'] = 0
    data.loc[data[config.split_column] == val_const, 'test'] = 0
    data.loc[data[config.split_column] == val_const, 'val'] = 1
    # check for overlapping patients in test and train/val sets
    if not(set(data.loc[data[config.split_column]==test_const, 'patient_id'].values).isdisjoint(set(data.loc[data[config.split_column]!=test_const, 'patient_id']))):
        # remove patients
        s=sum(data['train'].values)
        patients = set(data.loc[data[config.split_column]==test_const, 'patient_id']).intersection(set(data.loc[data[config.split_column]!=test_const, 'patient_id']))
        #print('size {:d}'.format(len(patients)))
        #print(data.loc[data['patient_id'].isin(list(patients))&data['train']==1].shape[0])
        data.loc[(data['patient_id'].isin(list(patients)))&(data[config.split_column]!=test_const), 'train']=0
        data.loc[(data['patient_id'].isin(list(patients))) & (data[config.split_column]!=test_const), 'val'] = 0
        print('Removed {:d} entries from the training and validation sets because the patients appeared in the test set'.format(s-sum(data['train'].values)))    
    
    if not(set(data.loc[data[config.split_column]==val_const, 'patient_id'].values).isdisjoint(set(data.loc[data[config.split_column]!=val_const, 'patient_id']))):
        # remove patients
        s=sum(data['train'].values)
        patients = set(data.loc[data[config.split_column]==val_const, 'patient_id']).intersection(set(data.loc[data[config.split_column]<val_const, 'patient_id']))
        data.loc[(data['patient_id'].isin(list(patients)))&(data[config.split_column]!=val_const), 'train']=0
        print('Removed {:d} entries from the training set because the patients appeared in the validation set'.format(s-sum(data['train'].values)))

    train_size = data.loc[data['train']==1].shape[0]
    val_size = data.loc[data['val'] ==1].shape[0]
    test_size = data.loc[data['test']==1].shape[0]
   
    print('Train set size = {train}, val set size = {val}, test set size = {test}'.format(train=train_size,  val=val_size, test=test_size))   
    return data


def transform(data):
    # convert length of stay feature
    # 1 - more then 7 days, 0 - less
    data["los"]=data["los"].apply(binary_legth_of_stay)
    data["hospital_id"] = data["hospital_id"].replace(HOSPITAL_ID)
    data  = transform_diagnosis(data)
    return data

