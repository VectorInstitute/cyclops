import getpass
import pandas as pd
import numpy as np
import pandas as pd
import json
import os
import time
import sqlalchemy 
import pandas.io.sql as psql
import argparse

def read_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--user", default='koshkinam', type=str, required=False, help='Postgres user')
    # parser.add_argument("--password", default=os.environ['PGPASSWORD'], type=str, required=True, help='Postgres password')
    parser.add_argument("--port", default=5432, type=int, help='Postgres port')
    parser.add_argument("--host", default='db.gemini-hpc.ca', type=str, required=False, help='Postgres host')
    parser.add_argument("--database", default='delirium_v3_0_0', type=str, required=False, help='Postgres database')
    parser.add_argument("--output", type=str, required=True, help='Where should we put the CSV results?')

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
      LIMIT 10"""

    data=pd.read_sql(query ,con=engine)
    print(data.head())
    print(data.columns.tolist())

    return data

def binary_legth_of_stay(l):
    return 1 if l >= 7 else 0    

def transform(data):
    # convert length of stay feature
    # 1 - more then 7 days, 0 - less
    data["stay_length"]=data["stay_length"].apply(binary_legth_of_stay)
    return data

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
    save_to_disk(data, config)

