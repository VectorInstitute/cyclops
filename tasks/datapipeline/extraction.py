
import pandas as pd
import numpy as np
import json
import os
import time
import sqlalchemy
import pandas.io.sql as psql
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
    #print(data.head())

    #temp: test labs:
    config.min_percent  = 0
    config.aggregation_window = 6
    config.output = config.output_folder
    labs(config, engine)

    return data

def labs(args, engine):
    idx = pd.IndexSlice
    query_1 = "SELECT REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ') as unique_lab_names, COUNT(REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ')) as unique_lab_counts FROM lab GROUP BY unique_lab_names ORDER BY unique_lab_counts ASC;"  # select all lab.test_name_raw (equivalent to itemids in mimiciii)

    df = pd.read_sql(query_1, con=engine)
    display(df.head(10))
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

