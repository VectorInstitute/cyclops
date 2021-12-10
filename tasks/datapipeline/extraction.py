
import pandas as pd
import numpy as np
import json
import os
import time
import sqlalchemy
import pandas.io.sql as psql
import re
from tqdm import tqdm

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

DRUG_SCREEN=['amitriptyline', 'amphetamine', 'barbiturates', 'barbiturates_scn', 'barbiturates_and_sedatives_blood',
                      'benzodiazepine_scn', 'benzodiazepines_screen', 'cannabinoids',
                     'clozapine', 'cocaine', 'cocaine_metabolite', 'codeine', 'cocaine_metabolite',
                     'codeine_metabolite_urine', 'desipramine', 'dextromethorphan', 'dim_per_dip_metabolite',
                     'dimen_per_diphenhydramine', 'doxepin', 'ephedrine_per_pseudo', 'fluoxetine',
                     'hydrocodone', 'hydromorphone', 'imipramine', 'lidocaine', 'mda_urine', 'mdma_ecstacy',
                     'methadone', 'meperidine_urine', 'methadone_metabolite_urine', 'methamphetamine', 'morphine',
                     'morphine_metabolite_urine', 'nortriptyline', 'olanzapine_metabolite_u',
                     'olanzapine_urine', 'opiates_urine', 'oxycodone', 'oxycodone_cobas', 'oxycodone_metabolite',
                      'phenylpropanolamine', 'propoxyphene', 'sertraline',
                     'trazodone', 'trazodone_metabolite', 'tricyclics_scn', 'venlafaxine', 'venlafaxine_metabolite']

def numeric_categorical(item):
    x=None
    locals_=locals()
    
    item = '('+item.replace('to', ' + ').replace('-', ' + ')+')/2' # this neglects negative ranges. todo, find a fast regex filter
    items = item.replace('  ',' ').replace('(', '( ').split(' ')
    item=" ".join([i.lstrip('0') for i in items])
    exec('x='+item, globals(), locals_)
    return locals_['x']


def x_to_numeric(x):
    """
    Handle different strings to convert to numeric values(which can't be done in psql)
    """
    if x is None:
        return np.nan
    elif x is np.nan:
        return np.nan
    elif isinstance(x, str):
        # check if it matches the categorical pattern "2 to 5" or 2 - 5
        if re.search(r'-?\d+ +(to|-) +-?\d+', x) is not None:
            try:
                return numeric_categorical(x)
            except:
                print(x)
                raise
        return re.sub('^-?[^0-9.]', '', str(x))
        
def name_count(items):
    """
    Inputs:
        items (list): a list of itemsto count the number of repeating values
    Returns:
        list of tuples with the name of the occurence and the count of each occurence
    """
    all_items={}
    for item in items:
        if item in all_items.keys():
            all_items[item]+=1
        else:
            all_items[item]=1
    return sorted([(k,v) for k,v in all_items.items()], key=lambda x:-x[1])

def get_scale(be_like, actual):
    """
    This function is applied to scale every measurement to a standard unit
    """
    replacements={'milliliters':'ml',
            'millimeters':'mm',
            'gm':'g',
            'x10 ':'x10e',
            'tril':'x10e12',
            'bil':'x10e9'
            }
    if isinstance(be_like, str) & isinstance(actual, str):
        # check if anything should be replaced:
        for k, v in replacements.items():
            be_like = be_like.replace(k,v)
            actual = actual.replace(k, v)
        scale=1
        
        # check if both have x10^X terms in them
        multipliers = ['x10e6', 'x10e9', 'x10e12']
        if any(item in be_like for item in multipliers) and any(item in actual for item in multipliers):
            # then adjust
            scale*=1000**-multipliers.index(re.search('x10e\d+', actual)[0]) * 1000**multipliers.index(re.search('x10e\d+', be_like)[0])
            return scale
        
        
        be_like_list=be_like.split('/') # split the numerator and denominators for the comparator units
        actual_list=actual.split('/')# split the numerator and denominators for the units that need to be converted
        if len(be_like_list) == len(actual_list):
            success=1
            for i in range(len(be_like_list)):
                try:
                    scale*=convert(actual_list[i], be_like_list[i])**(1 if i>0 else -1)
                except:
                    success=0
                    # could not convert between units
                    break
            if success: return scale
    return 'could not convert'

def filter_string(item):
    item=item.replace(')', ' ')
    item=item.replace('(', ' ')
    item=item.replace('%', ' percent ')
    item=item.replace('+', ' plus ')
    item=item.replace('#', ' number ')
    item=item.replace('&', ' and ')
    item=item.replace("'s", '')
    item=item.replace(',', ' ')
    item=item.replace('/', ' per ')
    item=' '.join(item.split())
    item=item.strip()
    
    
    item=item.replace(' ', '_')
    item=re.sub('[^0-9a-z_()]+', '_', item)
    if len(item)>1:
        if item[0] in '1234567890_':
            item='a_'+item
    return item


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
    #print(data.head())

    #temp: test labs:
    config.min_percent  = 0
    config.aggregation_window = 6
    config.output = config.output_folder
    labs(config, engine)

    return data


def simple_imput(mean_vals):
    """
    """
    idx = pd.IndexSlice
    # do simple imputation
    # mask
    mask = 1 - mean_vals.isna()
    # measurement
    measurement = mean_vals.copy()

    print(measurement.loc[idx[:, :, 0], :].head())
    print(measurement.loc[idx[:, :, 0], :].values.shape)
    print(measurement.mean().values.shape)

    print(measurement.loc[idx[:, :, 0, :]].groupby('genc_id').count())

    # these expressions necessarily need to be executed seperately
    subset_data = measurement.loc[idx[:, :, 0], :]
    data_means = measurement.mean()
    subset_data = subset_data.fillna(data_means)
    measurement.loc[idx[:, :, 0], :] = subset_data.values

    measurement = measurement.ffill()
    # time_since
    is_absent = 1 - mask
    hours_of_absence = is_absent.groupby(['patient_id', 'genc_id']).cumsum()
    time_df = hours_of_absence - hours_of_absence[is_absent == 0].fillna(method='ffill')
    time_df = time_df.fillna(0)

    final_data = pd.concat([measurement, mask, time_df], keys=['measurement', 'mask', 'time'], axis=1)
    final_data.columns = final_data.columns.swaplevel(0, 1)
    final_data.sort_index(axis='columns', inplace=True)

    nancols = 0

    try:
        nancols = np.sum([a == 0 for a in final_data.loc[:, idx[:, 'mask']].sum().values])
        print(nancols)
    except:
        print('could not get nancols')
        pass

    print(nancols, '/', len(sorted(set(final_data.columns.get_level_values(0)))))
    return final_data


def labs(args, engine):
    idx = pd.IndexSlice
    query_1 = "SELECT REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ') as unique_lab_names, COUNT(REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ')) as unique_lab_counts FROM lab GROUP BY unique_lab_names ORDER BY unique_lab_counts ASC;"  # select all lab.test_name_raw (equivalent to itemids in mimiciii)

    df = pd.read_sql(query_1, con=engine)
    print(df.head(10))
    unique_items = df['unique_lab_names'].values

    print('Total number of labs: ', len(df))
    print('\t num labs:  ', (df['unique_lab_counts'].sum()))
    print('Total number of labsmeasured more than once: ', len(df.loc[df['unique_lab_counts'] > 1, :]))
    print('\t num labs:  ', (df.loc[df['unique_lab_counts'] > 1, 'unique_lab_counts'].sum()))
    print('Total number of labsmeasured more than 10x: ', len(df.loc[df['unique_lab_counts'] >= 10, :]))
    print('\t num labs:  ', (df.loc[df['unique_lab_counts'] >= 10, 'unique_lab_counts'].sum()))
    print('Total number of labsmeasured more than 100x: ', len(df.loc[df['unique_lab_counts'] >= 100, :]))
    print('\t num labs:  ', (df.loc[df['unique_lab_counts'] >= 100, 'unique_lab_counts'].sum()))

    # we lose aboutr 20k observations

    unique_items = df.loc[df['unique_lab_counts'] >= 100, 'unique_lab_names'].values
    # todo, we sould apply min_percent instead.

    # min percent
    # get the count of unique participants

    if args.min_percent >= 1: args.min_percent = args.min_percent / 100

    print("appling a min_percent of {args.min_percent} results in")

    print('Total number of labsmeasured more than 100x: ',
          len(df.loc[df['unique_lab_counts'] / len(df) >= args.min_percent, :]))

    print("\t num_labs:  ", df.loc[df['unique_lab_counts'] / len(df) >= args.min_percent, 'unique_lab_counts'].sum())
    # unique_items = df.loc[df['unique_lab_counts']/len(df)>= args.min_percent, 'unique_lab_names' ].values

    # input()

    print('Warning min_percent is not applied')

    # get the hospitals in the dataframes
    query_2 = "SELECT DISTINCT hospital_id FROM ip_administrative;"

    dataset_hospitals = pd.read_sql(query_2, con=engine)

    print(dataset_hospitals)
    print(dataset_hospitals.values.ravel())

    return_dfs = {}

    for site in dataset_hospitals.values.ravel():
        print(site)
        assert site in HOSPITAL_ID.keys(), f"Could not find site {site} in constants.py HOSPITAL_ID dict"
        #

        # add the case strings (all of the columns)
        query = f"""SELECT a.*,
            a.test_name_raw_clean,
            a.result_unit_clean,
            CASE WHEN a.result_value_clean <> '.' THEN a.result_value_clean ELSE null END as result_value_clean,
            i.hospital_id,
            i.patient_id_hashed as patient_id,
            DATE_PART('day', a.sample_collection_date_time - i.admit_date_time)*24 + DATE_PART('hour', a.sample_collection_date_time - i.admit_date_time) AS hours_in
            FROM (select genc_id,
                sample_collection_date_time,
                CASE WHEN LOWER(lab.result_value) LIKE ANY('{{neg%%, not det%%,no,none seen, arterial, np}}') THEN '0'
                    WHEN LOWER(lab.result_value) LIKE ANY('{{pos%%, det%%, yes, venous, present}}') THEN '1'
                    WHEN LOWER(lab.result_value) = ANY('{{small, slight}}') THEN '1'
                    WHEN LOWER(lab.result_value) = 'moderate' THEN '2'
                    WHEN LOWER(lab.result_value) = 'large' THEN '3'
                    WHEN LOWER(lab.result_value) = 'clear' THEN '0'
                    WHEN LOWER(lab.result_value) = ANY('{{hazy, slcloudy, mild}}') THEN '1'
                    WHEN LOWER(lab.result_value) = ANY('{{turbid, cloudy}}') THEN '2'
                    WHEN LOWER(lab.result_value) = 'non-reactive' THEN '0'
                    WHEN LOWER(lab.result_value) = 'low reactive' THEN '1'
                    WHEN LOWER(lab.result_value) = 'reactive' THEN '2'
                    WHEN REPLACE(lab.result_value, ' ', '') ~ '^(<|>)?=?-?[0-9]+\.?[0-9]*$'  THEN substring(lab.result_value from '(-?[0-9.]+)')
                    WHEN lab.result_value ~ '^[0-9]{1}\+'  THEN substring(lab.result_value from '([0-9])')
                    WHEN lab.result_value ~ '^-?\d+ +(to|-) +-?\d+' THEN substring(lab.result_value from '(-?\d+ +(to|-) +-?\d+)')
                ELSE null END AS result_value_clean,
                LOWER(lab.result_unit) as result_unit_clean,
                REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ') AS test_name_raw_clean
                FROM lab ) a

            LEFT OUTER JOIN (SELECT ip_administrative.patient_id_hashed,
                ip_administrative.genc_id,
                ip_administrative.admit_date_time,
                ip_administrative.hospital_id
                FROM ip_administrative) i
            ON a.genc_id=i.genc_id   
            ORDER BY patient_id, a.genc_id ASC, hours_in ASC
            LIMIT 25000000;
            """

        query = f"""SELECT a.genc_id,
            CASE WHEN LOWER(a.result_value) LIKE ANY('{{neg%%, not det%%,no,none seen, arterial, np}}') THEN '0'
                WHEN LOWER(a.result_value) LIKE ANY('{{pos%%, det%%, yes, venous, present}}') THEN '1'
                WHEN LOWER(a.result_value) = ANY('{{small, slight}}') THEN '1'
                WHEN LOWER(a.result_value) = 'moderate' THEN '2'
                WHEN LOWER(a.result_value) = 'large' THEN '3'
                WHEN LOWER(a.result_value) = 'clear' THEN '0'
                WHEN LOWER(a.result_value) = ANY('{{hazy, slcloudy, mild}}') THEN '1'
                WHEN LOWER(a.result_value) = ANY('{{turbid, cloudy}}') THEN '2'
                WHEN LOWER(a.result_value) = 'non-reactive' THEN '0'
                WHEN LOWER(a.result_value) = 'low reactive' THEN '1'
                WHEN LOWER(a.result_value) = 'reactive' THEN '2'
                WHEN REPLACE(a.result_value, ' ', '') ~ '^(<|>)?=?-?[0-9]+\.?[0-9]*$'  THEN substring(a.result_value from '(-?[0-9.]+)')
                WHEN a.result_value ~ '^[0-9]{1}\+'  THEN substring(a.result_value from '([0-9])')
                WHEN a.result_value ~ '^-?\d+ +(to|-) +-?\d+' THEN substring(a.result_value from '(-?\d+ +(to|-) +-?\d+)')
            ELSE null END AS result_value_clean,
            LOWER(a.result_unit) as result_unit_clean,
            --CASE WHEN a.result_value_clean <> '.' THEN a.result_value_clean ELSE null END as result_value_clean,
            i.hospital_id,
            i.patient_id_hashed as patient_id,
            DATE_PART('day', a.sample_collection_date_time - i.admit_date_time)*24 + DATE_PART('hour', a.sample_collection_date_time - i.admit_date_time) AS hours_in,
            REPLACE(REPLACE(LOWER(a.lab_test_name_raw),'.','') , '-', ' ') AS test_name_raw_clean
            FROM lab a
            INNER JOIN ip_administrative i
            ON a.genc_id=i.genc_id
            WHERE i.hospital_id='{site}'
            ORDER BY patient_id, a.genc_id ASC, hours_in ASC;
            """

        with open(f'{site}_lab_query3.sql', 'w') as f:
            f.write(query)

        do_load = True  # set to false to reload the query (takes ~ 20 minutes for 3000 patients)
        if os.path.exists(os.path.join(args.output, f'{site}_tmp_labs2.csv')) and do_load:
            print('loading from disk')
            df = pd.read_csv(os.path.join(args.output, f'{site}_tmp_labs2.csv'))
            df.set_index(['patient_id', 'genc_id', 'hours_in'], inplace=True)
        else:
            print('constructing from scratch')
            time_old = time.time()
            df = pd.read_sql(query, con=engine)
            print(time.time() - time_old)
            # drop the duplicated genc_id column
            df = df.loc[:, ~df.columns.duplicated()]

            df['hours_in'] = (df['hours_in'].values / args.aggregation_window).astype(int) * int(
                args.aggregation_window)
            df.set_index(['patient_id', 'genc_id', 'hours_in'], inplace=True)
            # Unstack columns
            print('successfully queried, now unstacking columns')

            # replace the drug_screen values with the True/False.

            print("TODO: convert DRUG SCREEN to categorical yes/no")
            print(set(df.loc[df['test_name_raw_clean'].isin(DRUG_SCREEN), 'result_value_clean'].values.tolist()))

            print(df.loc[df['test_name_raw_clean'].isin(DRUG_SCREEN), ['result_value_clean',
                                                                       'test_name_raw_clean']].groupby(
                'result_value_clean').count())

            print(df.columns.tolist())

            df.loc[~df['result_value_clean'].isna(), 'result_value_clean'] = df.loc[
                ~df['result_value_clean'].isna(), 'result_value_clean'].apply(x_to_numeric)

            df['result_value_clean'] = pd.to_numeric(df['result_value_clean'], 'coerce')

            # standardise the units

            # for col in df.columns.tolist():
            #     print(col)

            # #filter values #'test_name_raw_clean'
            # to_remove=['lab_test_name_raw', 'test_code_raw', 'test_type_mapped', 'result_value', 'result_value_clean', 'reference_range', 'sample_collection_date_time', 'row_id']
            # try:
            #     df.drop(to_remove, axis=1, inplace=True)
            # except KeyError:
            #     pass

            # sort out all of the drug screen cols to be True/False
            for col in tqdm(DRUG_SCREEN, desc='drug_screen'):
                df.loc[(df['test_name_raw_clean'].isin(DRUG_SCREEN)) & (
                            (~df['result_unit_clean'].isna()) & (~df['result_value_clean'].isna())) & (
                                   df['result_value_clean'] > 0), 'result_value_clean'] = 1.0
                # make the unit NaN
                df.loc[(df['test_name_raw_clean'].isin(DRUG_SCREEN)) & (~df['result_unit_clean'].isna()) & (
                    ~df['result_value_clean'].isna()), 'result_unit_clean'] = np.nan

            print(df.columns.tolist())
            df.reset_index().to_csv(os.path.join(args.output, f'{site}_tmp_labs2.csv'))

        # --------------------------------------------    UNIT CONVERSION -------------------------------------------------------------------

        units_dict = \
        df[['hospital_id', 'test_name_raw_clean', 'result_unit_clean']].groupby(['hospital_id', 'test_name_raw_clean'])[
            'result_unit_clean'].apply(name_count).to_dict()

        conversion_list = []

        df_cols = df.columns.tolist()
        for k, v in units_dict.items():
            if len(v) > 1:
                for item in v[1:]:
                    scale = get_scale(v[0][0], item[0])
                    if not (isinstance(scale, str)):
                        conversion_list.append((k[0], k[1], item[0], get_scale(v[0][0], item[0]), v[0][0],
                                                item[1]))  # key: (original unit, scale, to_unit)

        for item in conversion_list: print(item)

        try:
            print(f'rescued {np.sum(list(zip(*conversion_list))[-1])} labs by unit conversion')
        except:
            pass

        def clean(item_name):
            return str(item_name).replace('%', '%%').replace("'", "''"), filter_string(str(item_name))

        # apply the scaling we found to the df
        for item in tqdm(conversion_list, desc='unit_conversion'):
            hospital_id, lab_name, from_unit, scale, to_unit, count = item
            if clean(lab_name)[1] not in df_cols:
                print(clean(lab_name))
                continue
            assert lab_name == clean(lab_name), f"{lab_name} is not the same as {clean(lab_name)}"
            df.loc[(df['hospital_id'] == hospital_id) & (df['test_name_raw_clean'] == lab_name) & (
                        df['result_unit_clean'] == from_unit), clean(lab_name)[1]] *= scale
            df.loc[(df['hospital_id'] == hospital_id) & (df['test_name_raw_clean'] == lab_name) & (
                        df['result_unit_clean'] == from_unit), 'result_unit_clean'] = to_unit

        # --------------------------------------------    UNIT CONVERSION FOR UNKNOWN UNITS -------------------------------------------------------------------

        units_dict = \
        df[['hospital_id', 'test_name_raw_clean', 'result_unit_clean']].groupby(['hospital_id', 'test_name_raw_clean'])[
            'result_unit_clean'].apply(name_count).to_dict()

        log = ""
        determined_same = []
        df_cols = df.columns.tolist()
        for k, v in tqdm(units_dict.items()):
            if clean(k[1])[1] not in df_cols:
                continue
            if len(v) > 1:
                if v[0][1] < 20:
                    continue
                if isinstance(v[0][0], str):
                    # comparison = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean']==v[0][0]), clean(k[1])[1]].values
                    comparison = df.loc[(df['hospital_id'] == k[0]) & (df['test_name_raw_clean'] == k[1]) & (
                                df['result_unit_clean'] == v[0][0]), 'result_value_clean'].values
                elif np.isnan(v[0][0]):
                    # comparison = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean'].isna()), clean(k[1])[1]].values
                    comparison = df.loc[(df['hospital_id'] == k[0]) & (df['test_name_raw_clean'] == k[1]) & (
                        df['result_unit_clean'].isna()), 'result_value_clean'].values
                else:
                    raise
                norm_p = scipy.stats.kstest(comparison, 'norm')[1]

                for item in v[1:]:
                    if item[1] < 20:
                        #                 print(f'too few samples for {k[1]}, {v[0][0]}, {v[0][1]}, {item[0]}, {item[1]}')
                        continue
                    if isinstance(item[0], str):
                        # test_unit = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean']==item[0]), clean(k[1])[1]].values
                        test_unit = df.loc[(df['hospital_id'] == k[0]) & (df['test_name_raw_clean'] == k[1]) & (
                                    df['result_unit_clean'] == item[0]), 'result_value_clean'].values
                    elif np.isnan(item[0]):
                        # test_unit = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean'].isna()), clean(k[1])[1]].values
                        test_unit = df.loc[(df['hospital_id'] == k[0]) & (df['test_name_raw_clean'] == k[1]) & (
                            df['result_unit_clean'].isna()), 'result_value_clean'].values
                    else:
                        raise
                    same_test = scipy.stats.mannwhitneyu(comparison, test_unit)[1]
                    print(k[0], k[1], v[0][0], item[0], same_test)

                    log += f"Are {k[0]} and {k[1]} the same? units are {v[0][0]} and {item[0]}, respectively (p={same_test}) \r\n"
                    # sns.distplot([comparison, test_unit], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, label = [v[0][0], item[0]])

                    # input()
                    print((k[0], k[1], v[0][0], item[0], norm_p, same_test))
                    # if same_test >=0.05:
                    #     print((k[0], k[1], v[0][0], item[0], norm_p, same_test))
                    #     if input()=='':
                    #         determined_same.append((k[0], k[1], v[0][0], item[0], item[1], norm_p, same_test))

        # put log to txt
        with open(os.path.join(args.output, site + '_units_log.txt'), 'w') as f:
            f.write(log)

        # print(determined_same)
        if len(determined_same) > 0:
            def clean(item_name):
                return str(item_name).replace('%', '%%').replace("'", "''"), filter_string(str(item_name))

            # print(print(sorted(df.columns.tolist())
            for item in tqdm(determined_same):
                print(item)
                hospital_id, lab_name, to_unit, from_unit, num_saved, norm_p, same_test = item
                if clean(lab_name)[1] not in df_cols:
                    print(clean(lab_name))
                    continue
                if isinstance(from_unit, str):
                    df.loc[(df['hospital_id'] == hospital_id) & (df['test_name_raw_clean'] == lab_name) & (
                                df['result_unit_clean'] == from_unit), 'result_unit_clean'] = to_unit
                else:
                    df.loc[(df['hospital_id'] == hospital_id) & (df['test_name_raw_clean'] == lab_name) & (
                        df['result_unit_clean'].isna()), 'result_unit_clean'] = to_unit

            np.sum(list(zip(*determined_same))[4])

        units_dict = \
        df[['hospital_id', 'test_name_raw_clean', 'result_unit_clean']].groupby(['hospital_id', 'test_name_raw_clean'])[
            'result_unit_clean'].apply(name_count).to_dict()

        # eliminate features with unmatchable units

        keep_vars = []
        keep_vars_nan = []
        for k, v in units_dict.items():
            keep_vars.append((k[0], k[1], v[0][0]))

        print(len(df))
        print(sum(pd.Series(list(zip(df['hospital_id'], df['test_name_raw_clean'], df['result_unit_clean']))).isin(
            keep_vars)) / len(df))
        index = pd.Series(list(zip(df['hospital_id'], df['test_name_raw_clean'], df['result_unit_clean']))).isin(
            keep_vars).values

        df = df.loc[index]

        # drop the text fields prior to the groupby
        # if 'result_unit' in df.columns.tolist():
        #     df.drop(['result_unit', 'result_unit_clean', 'test_name_raw_clean'], axis=1, inplace=True)

        assert 'hospital_id' in df.columns

        # sometimes there are multiple hospitals. Let us just make sure that the first hospital is there for all of the patient's encounters
        df = df.drop(labels='hospital_id', axis=1).join(df.groupby(['patient_id', 'genc_id'])['hospital_id'].first(),
                                                        how='left', on=['patient_id', 'genc_id'])

        df['hospital_id'] = df['hospital_id'].replace(HOSPITAL_ID)

        df.reset_index('hours_in', inplace=True)
        df['hours_in'] = df['hours_in'].apply(lambda x: min(0, x))
        df.set_index('hours_in', append=True, inplace=True)
        print("doing massive groupby")
        mean_vals = df.groupby(['patient_id', 'genc_id', 'hours_in', 'hospital_id', 'test_name_raw_clean',
                                'result_unit_clean']).mean()  # could also do .agg('mean', 'median', 'std', 'min', 'max') for example.

        print("doing massive unstack")

        mean_vals = mean_vals.unstack(level=['test_name_raw_clean', 'result_unit_clean'])

        # potentially drop oth level of multiindex
        mean_vals = mean_vals.droplevel(level=0, axis='columns')

        assert 'hospital_id' in mean_vals.index.names

        mean_vals = mean_vals.groupby(by=mean_vals.columns,
                                      axis=1).mean()  # because we made many of the columns the same

        print(mean_vals.head())

        print(len(mean_vals), ' patients had ',
              len(mean_vals.columns) * len(set(mean_vals.index.get_level_values('hospital_id'))), ' lab types from ',
              len(set(mean_vals.index.get_level_values('hospital_id'))), ' hospitals.')

        # Now eliminate rows with fewer than args.min_percent data
        drop_cols = []
        # for name, hospital_id in HOSPITAL_ID.items():
        #     # only get participants that don't have all np.nan for hospital id==hospital_id
        #     temp_df = mean_vals.loc[mean_vals.index.get_level_values('hospital_id')==hospital_id]
        #     temp_df = temp_df.unstack( level='hospital_id')
        mean_vals = mean_vals.dropna(axis='columns', how='all')
        # print(name, hospital_id, len(temp_df))

        for col in mean_vals.columns:
            print("obs_rate for ", col, (1 - mean_vals[col].isna().mean()))
            if (1 - mean_vals[col].isna().mean()) < args.min_percent:
                print(col, 1 - mean_vals[col].isna().mean())
                # drop the column.
                drop_cols.append(col)

        # mean_vals['hospital_id2']= mean_vals.index.get_level_values('hospital_id')
        # mean_vals = mean_vals.set_index('hospital_id2', append=True)
        # mean_vals = mean_vals.unstack('hospital_id2')

        # mean_vals.columns.names = [n.replace('_id2', '_id') for n in mean_vals.columns.names]
        # mean_vals = mean_vals.set_index('hospital_id2', append=True)

        print(mean_vals.head())

        mean_vals = mean_vals.drop(drop_cols, axis=1)

        print("After applying args.min_percent==", args.min_percent)
        print(len(mean_vals), ' patients had ', len(mean_vals.columns), ' lab types from ', site)

        mean_vals.sort_index(inplace=True)

        return_dfs.update({site: mean_vals})
        # break #TODO stop this

        ########## SITE ITERATOR

        for site, mean_vals in return_dfs.items():
            #outcomes_df = outcomes_df_master.copy()

            # get the overlapping genc_ids:
            #print('outcomes_genc_ids across all sites:', len(set(outcomes_df.index.get_level_values('genc_id'))))
            print(f'genc_ids in data at {site}:', len(set(mean_vals.index.get_level_values('genc_id'))))

            #merged_genc_ids = set(outcomes_df.index.get_level_values('genc_id')).intersection(
            #    set(mean_vals.index.get_level_values('genc_id')))

            #print('combined:', len(merged_genc_ids))

            # this excludes some people that
            #outcomes_df = outcomes_df.loc[outcomes_df.index.get_level_values('genc_id').isin(merged_genc_ids)]
            #mean_vals = mean_vals.loc[mean_vals.index.get_level_values('genc_id').isin(merged_genc_ids)]

            # first assert that all genc_ids are in the outcomes from the mean_vals site.
            mean_vals_genc_ids = set(mean_vals.index.get_level_values('genc_id'))
            # for m in mean_vals_genc_ids:
            #     if m not in outcomes_df.index.get_level_values('genc_id'):
            #         print(m)
            #assert (all([m in outcomes_df.index.get_level_values('genc_id') for m in
            #             mean_vals_genc_ids]))  # everyone has a discharge, transfer, or death.

            # but not all patients in outcomes have mean_vals

            #outcomes_df = outcomes_df.loc[
            #    outcomes_df.index.get_level_values('genc_id').isin(mean_vals.index.get_level_values('genc_id'))]

            # get the common index of only genc_id, hours_in
            mean_vals.reset_index(['patient_id', 'hospital_id'], inplace=True)
            #outcomes_df.reset_index(['patient_id', 'hospital_id'], inplace=True)
            #joined_index = mean_vals.index.union(outcomes_df.index)
            #print('len(mean_vals) + len(outcomes_df) = len(mean_vals)+len(outcomes_df); len(joined_index)')
            #print(f'{len(mean_vals)} + {len(outcomes_df)} = {len(mean_vals) + len(outcomes_df)}; {len(joined_index)}')
            print('data: ', len(set(mean_vals.index.get_level_values('genc_id'))))
            #print('outcomes: ', len(set(outcomes_df.index.get_level_values('genc_id'))))

            #print(len(set(outcomes_df.reset_index('hours_in').index)),
            #      len(set(mean_vals.reset_index('hours_in').index)))
            #print(len(set(outcomes_df.reset_index('hours_in').index.union(mean_vals.reset_index('hours_in').index))))

            # Instead of just having the end index, we want the entire length od stay regularly spaced.
            #genc_min_max = pd.DataFrame(index=joined_index).reset_index()[['genc_id', 'hours_in']].groupby(
            #    'genc_id').agg({'hours_in': ['min', 'max']})  # tuple of (genc_id, min_index, max_index)
            #print(genc_min_max.head())
            #genc_min_max = genc_min_max.set_index([('hours_in', 'min'), ('hours_in', 'max')],
            #                                      append=True).index.tolist()
            # now for each tuple, fill between min_index and max_index # TODO: how does this compare to the args aggregation window?
            #new_index = []
            #for item in genc_min_max:
            #    # genc_id, min_hour, max_hour = item
            #    hours_range = list(range(item[1], item[2] + 1))
            #    new_index += list(zip([item[0]] * len(hours_range), hours_range))
            #new_index = pd.MultiIndex.from_tuples(new_index, names=['genc_id', 'hours_in'])

            # reindex dataframes
            try:
                # make sure it is only ['genc_id', 'hours_in]
                #outcomes_df.reset_index(['patient_id', 'hospital_id'], inplace=True)
                mean_vals.reset_index(['patient_id', 'hospital_id'], inplace=True)
            except:
                #assert len(outcomes_df.index.names) == 2
                assert len(mean_vals.index.names) == 2
                #assert len(set(outcomes_df.index.names).intersection({'genc_id', 'hours_in'})) == 2
                #assert len(set(outcomes_df.index.names).intersection({'genc_id', 'hours_in'})) == 2

            #print(outcomes_df.head())
            print(mean_vals.head())

            #outcomes_df = outcomes_df.replace('', np.nan)

            # print(outcomes_df['patient_id'].isna().sum())
            #outcomes_df = outcomes_df.reindex(new_index)
            # now ffill the patient_id and hospital_id for all the new hours in we just added
            # print(outcomes_df['patient_id'].isna().sum())
            #outcomes_df[['patient_id', 'hospital_id']] = outcomes_df[['patient_id', 'hospital_id']].groupby('genc_id')[
            #    ['patient_id', 'hospital_id']].ffill().groupby('genc_id')[['patient_id', 'hospital_id']].bfill()
            # print(outcomes_df['patient_id'].isna().sum())
            #outcomes_df = outcomes_df.reset_index().set_index(['patient_id', 'genc_id', 'hours_in', 'hospital_id'])

            #mean_vals = mean_vals.reindex(new_index)
            # now ffill the patient_id and hospital_id for all the new hours in we just added
            # print('mean_vals: ',mean_vals['patient_id'].isna().sum())
            mean_vals[['patient_id', 'hospital_id']] = \
            mean_vals[['patient_id', 'hospital_id']].groupby('genc_id')[['patient_id', 'hospital_id']].ffill().groupby(
                'genc_id')[['patient_id', 'hospital_id']].bfill()
            # print(mean_vals['patient_id'].isna().sum())
            mean_vals = mean_vals.reset_index().set_index(['patient_id', 'genc_id', 'hours_in', 'hospital_id'])

            # assert that ll genc_ids have exactly 1 unique patient id (excluding NaN)

            # assert all([i<=1 for i in outcomes_df.reset_index().groupby('genc_id')['patient_id'].nunique().values]) # some patients do not have patient IDs if they didn't have health cards.

            # print(outcomes_df.reset_index()['patient_id'].isna().sum())
            # print(outcomes_df.reset_index()['patient_id'].isin(['']).sum())
            # print(len(outcomes_df.reset_index()['patient_id'].isna()))
            # print(len(outcomes_df.reset_index()['patient_id'].isin([''])))

            # print(len(outcomes_df.loc[(outcomes_df.reset_index()['patient_id'].isna())|(outcomes_df.reset_index()['patient_id'].isin(['']))]))
            # genc_ids = outcomes_df.loc[(outcomes_df.reset_index()['patient_id'].isna())|(outcomes_df.reset_index()['patient_id'].isin(['']))].index.get_level_values('genc_id')

            # print(outcome_df.loc[outcome_df.index.get_level_values('genc_id').isin(genc_ids)])

            #assert all([i == 1 for i in outcomes_df.reset_index().groupby('genc_id')['patient_id'].nunique().values])

            # assert that there are no missing patient ids or hospital ids
            # assert np.isnan(outcomes_df.index.get_level_values('patient_id')).sum()==0
            #assert np.isnan(outcomes_df.index.get_level_values('hospital_id')).sum() == 0
            # assert np.isnan(mean_vals.index.get_level_values('patient_id')).sum()==0
            assert np.isnan(mean_vals.index.get_level_values('hospital_id')).sum() == 0

            # forward fill outcomes_df
            #idx = pd.IndexSlice
            #min_index = outcomes_df.reset_index('hours_in').groupby(
            #   ['patient_id', 'genc_id', 'hospital_id']).min().set_index('hours_in', append=True).swaplevel(
            #    i='hours_in', j='hospital_id').index  # back to patient_id, genc_id, hours_in, hospital_id
            # outcomes_df.loc[idx[:, :, 0], :]=outcomes_df.loc[idx[:, :, 0], :].fillna(0)
            #outcomes_df.loc[min_index, :] = outcomes_df.loc[min_index, :].fillna(
            #    0)  # fill all the minimum values with 0
            # first the resuscitation columns must be forward filled witha  limit

            # currently aggregation window isn't applied yet.
            #print(outcomes_df.head())
            #outcomes_df['resus_24'] = outcomes_df['resus_24'].ffill(limit=24).fillna(0)
            #outcomes_df['resus_48'] = outcomes_df['resus_24'].ffill(limit=48).fillna(0)
            # ffill the rest of the outcomes.
            #outcomes_df = outcomes_df.ffill()

            # do we want to add a gap time at the end for outcomes to prevent leakage?

            # formerly the groups were already hourly, but now we need to aggregate.
            # mean_vals.index=mean_vals.index.set_levels(mean_vals.index.levels[2]*args.aggregation_window, level=2)
            # outcomes_df.index=outcomes_df.index.set_levels(outcomes_df.index.levels[2]*args.aggregation_window, level=2)

            # time at zero is a groupby mean
            mean_vals_0 = mean_vals.loc[mean_vals.index.get_level_values('hours_in') < 0, :].groupby(['patient_id', 'genc_id', 'hospital_id']).mean()

            mean_vals_0['hours_in'] = 0
            mean_vals_0 = mean_vals_0.reset_index().set_index(['patient_id', 'genc_id', 'hours_in', 'hospital_id'])

            # now make sure mean_vals is 0 and up:
            mean_vals = mean_vals.loc[mean_vals.index.get_level_values('hours_in') >= 0, :]

            print(mean_vals_0.head())
            # create new column which is hours in aggregator
            mean_vals['agg_hours_in'] = (mean_vals.index.get_level_values('hours_in') // int(
                args.aggregation_window) + 1) * int(args.aggregation_window)
            mean_vals = mean_vals.groupby(['patient_id', 'genc_id', 'agg_hours_in', 'hospital_id']).mean()
            mean_vals.index.names = ['patient_id', 'genc_id', 'hours_in', 'hospital_id']
            mean_vals = mean_vals.append(mean_vals_0).sort_index()

            # time at zero is a groupby mean
            #outcomes_df_0 = outcomes_df.loc[outcomes_df.index.get_level_values('hours_in') < 0, :].groupby(
            #    ['patient_id', 'genc_id', 'hospital_id']).max()

            #outcomes_df_0['hours_in'] = 0
            #outcomes_df_0 = outcomes_df_0.reset_index().set_index(['patient_id', 'genc_id', 'hours_in', 'hospital_id'])

            # now make sure mean_vals is 0 and up:
            #outcomes_df = outcomes_df.loc[outcomes_df.index.get_level_values('hours_in') >= 0, :]

            # create new column which is hours in aggregator
            #outcomes_df['agg_hours_in'] = (outcomes_df.index.get_level_values('hours_in') // int(
            #    args.aggregation_window) + 1) * int(args.aggregation_window)
            #outcomes_df = outcomes_df.groupby(['patient_id', 'genc_id', 'agg_hours_in', 'hospital_id']).max()
            #outcomes_df.index.names = ['patient_id', 'genc_id', 'hours_in', 'hospital_id']
            #outcomes_df = outcomes_df.append(outcomes_df_0).sort_index()

            # TODO add gap times
            # outcomes_df = add_gap_time(outcomes_df)

            final_data = simple_imput(mean_vals)

            # write out dataframes to hdf
            dynamic_hd5_filt_filename = 'all_hourly_data.h5'
            #outcomes_df.to_hdf(os.path.join(args.output, dynamic_hd5_filt_filename), f'interventions_{site}')
            final_data.to_hdf(os.path.join(args.output, dynamic_hd5_filt_filename), f'vitals_labs_{site}')

    return return_dfs


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
        test_const = int(config.test_split)
        val_const = int (config.val_split)    
    else:
        test_const = config.test_split
        val_const = config.val_split
    
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

