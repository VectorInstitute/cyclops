#import pyodbc
import getpass
import pandas as pd
import numpy as np
import pandas as pd
import json
import os
import time
import sqlalchemy 
import pandas.io.sql as psql


# Use your own username and whichever database you need
#server = 'db.gemini-hpc.ca' 
#database = 'delirium_v3_0_0' 
#username = 'koshkinam'
#password = getpass.getpass(prompt='Database password: ', stream=None) 
#con = pyodbc.connect('DRIVER={PostgreSQL};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
#ip_adm_df = pd.read_sql_query('SELECT * FROM diagnosis LIMIT 10',con=con)
#print(ip_adm_df.head())


def main(args):
    print('postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}')
    engine = sqlalchemy.create_engine(f'postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}')

    query = "select distinct
        i.patient_id_hashed as patient_id,
        i.genc_id,
        i.hospital_id,
        CASE when i.gender = 'F' THEN 1 ELSE 0 END AS sex,
        i.age,
        LOWER(i.country) as country,
	i.discharge_date_time, 
	i.admit_date_time,
        i.institution_from,
        i.institution_from_type,
        i.discharge_disposition::integer,
        i.institution_to,
        i.institution_to_type,
	i.province_territory_issuing_health_card_number as insurance,
     FROM ip_administrative i
     ORDER BY patient_id, genc_id
     LIMIT 10"

     data=pd.read_sql(query ,con=engine)
     print(data.head())

if __name__=="__main__":
    import argparse
    
    # read and write args
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--user", default='koshkinam', type=str, required=True, help='Postgres user')
    #parser.add_argument("--password", default=os.environ['PGPASSWORD'], type=str, required=True, help='Postgres password')
    parser.add_argument("--port", default=5432, type=int, help='Postgres port')
    parser.add_argument("--host", default='db.gemini-hpc.ca', type=str, required=True, help='Postgres host')
    parser.add_argument("--database", default='delirium_v3_0_0' , type=str, required=True, help='Postgres database')
    parser.add_argument("--output", type=str, required=True, help='Where should we put the CSV results?')

    args = parser.parse_args()

    args.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')

    print({k:v for k,v in vars(args).items()})

    password = getpass.getpass(prompt='Database password: ', stream=None) 
    arg.password = password
    
    # save args to args_{date}.json
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    print(date)

    with open(os.path.join(args.output, f'args_{date}.json'), 'w') as fp:
        fp.write(json.dumps({k:v for k,v in vars(args).items() if k!='password'}, indent=4))
    
    main(args)

