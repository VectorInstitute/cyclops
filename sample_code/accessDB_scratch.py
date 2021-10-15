import getpass
import pandas as pd
import numpy as np
import json
import os
import time
import sqlalchemy 
import pandas.io.sql as psql
import argparse

import plotly.offline as py #working offline
import plotly.graph_objs as go

from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab


def read_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--user", default='koshkinam', type=str, required=False, help='Postgres user')
    # parser.add_argument("--password", default=os.environ['PGPASSWORD'], type=str, required=True, help='Postgres password')
    parser.add_argument("--port", default=5432, type=int, help='Postgres port')
    parser.add_argument("--host", default='db.gemini-hpc.ca', type=str, required=False, help='Postgres host')
    parser.add_argument("--database", default='delirium_v3_0_0', type=str, required=False, help='Postgres database')
    parser.add_argument("--output", type=str, required=True, help='Where should we put the CSV results?')
    
    parser.add_argument('-w', action='store_true', help='Write extracted data to disk')

    args = parser.parse_args()

    # args.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')

    print({k: v for k, v in vars(args).items()})

    password = getpass.getpass(prompt='Database password: ', stream=None)
    args.password = password

    return args

def write_config(config):
    # save args to args_{date}.json
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    print(date)

    with open(os.path.join(config.output, f'args_{date}.json'), 'w') as fp:
        fp.write(json.dumps({k: v for k, v in vars(config).items() if k != 'password'}, indent=4))

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
    return 1 if l >= 7 else 0    

def transform(data):
    # convert length of stay feature
    # 1 - more then 7 days, 0 - less
    data["stay_length"]=data["stay_length"].apply(binary_legth_of_stay)
    print(data.columns.tolist())
    return data

#evaluate data drift with Evidently Profile
def eval_drift(reference, production, column_mapping, html=False):
    column_mapping['drift_conf_level'] = 0.95
    column_mapping['drift_features_share'] = 0.5
    data_drift_profile = Profile(sections=[DataDriftProfileSection])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    if html:
        dashboard = Dashboard(tabs=[DataDriftTab])
        dashboard.calculate(reference, production, column_mapping=column_mapping)
        dashboard.save("../data_drif_report.html")

    drifts = []
    for feature in column_mapping['numerical_features'] + column_mapping['categorical_features']:
        drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['p_value'])) 
    return drifts

def analyze(data, config):
    column_mapping = {}
    column_mapping['numerical_features'] = ['age']
    column_mapping['categorical_features'] = ['sex', 'stay_length']
    analysis_columns = column_mapping['numerical_features'] + column_mapping['categorical_features']
    reference_year = 2015 #TODO: change to use parameter
    evaluate_year = 2016 #TODO 
    reference_data = data.loc[data['year']==reference_year, analysis_columns]
    eval_data =  data.loc[data['year']==evaluate_year, analysis_columns]
    reference_data = reference_data.dropna()
    eval_data = eval_data.dropna()
    print(eval_data.head())    
    drifts = eval_drift (reference_data, eval_data, column_mapping, html=True)
    return drifts   

def save_to_disk(data, config, format='csv'):
    if (format != 'csv'):
        print("Unsupported format {}".format(format))
        exit
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    file_name = os.path.join(config.output, f'admin_data_{date}.csv')
    data.to_csv(file_name)


if __name__=="__main__":
    config = read_config()
    write_config(config)

    data = extract(config)
    data = transform(data)
    if config.w:
        save_to_disk(data, config)

    drifts = analyze(data, config)
   

