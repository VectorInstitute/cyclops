"""Data extraction module."""

import os
import time

import pandas as pd
import numpy as np
import scipy
import sqlalchemy
from tqdm import tqdm

from tasks.datapipeline.constants import (
    HOSPITAL_ID,
    DRUG_SCREEN,
)
from tasks.datapipeline.utils import (
    transform_diagnosis,
    name_count,
    clean,
    convert_units,
    x_to_numeric,
    simple_imput,
)


BASIC_DATA_ONLY = True


def extract(config):
    print(
        "postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
    )
    engine = sqlalchemy.create_engine(
        f"postgresql://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
    )

    pop_size = "" if config.pop_size == 0 else f"limit {config.pop_size}"
    filter = (
        f"WHERE DATE_PART('year', i.admit_date_time) <= {int(config.filter_year)}"
        if config.filter_year
        else ""
    )
    filter = (
        f"WHERE i.admit_date_time  > '{config.filter_date_from}' AND i.admit_date_time <= '{config.filter_date_to}'"
        if config.filter_date_from
        else filter
    )

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

    data = pd.read_sql(query_full, con=engine)
    #    data.set_index(['patient_id', 'genc_id'], inplace=True)

    if BASIC_DATA_ONLY:
        return data

    # temp: test labs:
    config.min_percent = 0
    config.aggregation_window = 6
    config.output = config.output_folder

    return_dfs = labs(config, engine)
    outcomes_df_master = outcomes(config, engine)

    ################################## Process/Combine results #####################################
    # adsd hospital id to index to match numerics
    print(set(outcomes_df_master["hospital_id"].replace(HOSPITAL_ID).values))
    outcomes_df_master["hospital_id"] = outcomes_df_master["hospital_id"].replace(
        HOSPITAL_ID
    )

    assert set(outcomes_df_master["hospital_id"].values).issubset(
        set(list(HOSPITAL_ID.keys()) + list(HOSPITAL_ID.values()) + ["", np.nan, None])
    )

    outcomes_df_master.set_index("hospital_id", append=True, inplace=True)

    ########## SITE ITERATOR

    for site, mean_vals in return_dfs.items():
        outcomes_df = outcomes_df_master.copy()

        # get the overlapping genc_ids:
        print(
            "outcomes_genc_ids across all sites:",
            len(set(outcomes_df.index.get_level_values("genc_id"))),
        )
        print(
            f"genc_ids in data at {site}:",
            len(set(mean_vals.index.get_level_values("genc_id"))),
        )

        merged_genc_ids = set(
            outcomes_df.index.get_level_values("genc_id")
        ).intersection(set(mean_vals.index.get_level_values("genc_id")))

        print("combined:", len(merged_genc_ids))

        # this excludes some people that
        outcomes_df = outcomes_df.loc[
            outcomes_df.index.get_level_values("genc_id").isin(merged_genc_ids)
        ]
        mean_vals = mean_vals.loc[
            mean_vals.index.get_level_values("genc_id").isin(merged_genc_ids)
        ]

        # first assert that all genc_ids are in the outcomes from the mean_vals site.
        mean_vals_genc_ids = set(mean_vals.index.get_level_values("genc_id"))
        assert all(
            [
                m in outcomes_df.index.get_level_values("genc_id")
                for m in mean_vals_genc_ids
            ]
        )  # everyone has a discharge, transfer, or death.

        # but not all patients in outcomes have mean_vals
        outcomes_df = outcomes_df.loc[
            outcomes_df.index.get_level_values("genc_id").isin(
                mean_vals.index.get_level_values("genc_id")
            )
        ]

        # get the common index of only genc_id, hours_in
        mean_vals.reset_index(["patient_id", "hospital_id"], inplace=True)
        outcomes_df.reset_index(["patient_id", "hospital_id"], inplace=True)
        joined_index = mean_vals.index.union(outcomes_df.index)
        print(
            "len(mean_vals) + len(outcomes_df) = len(mean_vals)+len(outcomes_df); len(joined_index)"
        )
        print(
            f"{len(mean_vals)} + {len(outcomes_df)} = {len(mean_vals) + len(outcomes_df)}; {len(joined_index)}"
        )
        print("data: ", len(set(mean_vals.index.get_level_values("genc_id"))))
        print("outcomes: ", len(set(outcomes_df.index.get_level_values("genc_id"))))

        print(
            len(set(outcomes_df.reset_index("hours_in").index)),
            len(set(mean_vals.reset_index("hours_in").index)),
        )
        print(
            len(
                set(
                    outcomes_df.reset_index("hours_in").index.union(
                        mean_vals.reset_index("hours_in").index
                    )
                )
            )
        )

        # Instead of just having the end index, we want the entire length od stay regularly spaced.
        genc_min_max = (
            pd.DataFrame(index=joined_index)
            .reset_index()[["genc_id", "hours_in"]]
            .groupby("genc_id")
            .agg({"hours_in": ["min", "max"]})
        )  # tuple of (genc_id, min_index, max_index)
        print(genc_min_max.head())
        genc_min_max = genc_min_max.set_index(
            [("hours_in", "min"), ("hours_in", "max")], append=True
        ).index.tolist()
        # now for each tuple, fill between min_index and max_index # TODO: how does this compare to the args aggregation window?
        new_index = []
        for item in genc_min_max:
            # genc_id, min_hour, max_hour = item
            hours_range = list(range(item[1], item[2] + 1))
            new_index += list(zip([item[0]] * len(hours_range), hours_range))
        new_index = pd.MultiIndex.from_tuples(new_index, names=["genc_id", "hours_in"])

        print("Before reindexing")
        print(outcomes_df.columns)
        print(mean_vals.columns)
        # reindex dataframes
        try:
            # make sure it is only ['genc_id', 'hours_in]
            outcomes_df.reset_index(["patient_id", "hospital_id"], inplace=True)
            mean_vals.reset_index(["patient_id", "hospital_id"], inplace=True)
        except:
            assert len(outcomes_df.index.names) == 2
            assert len(mean_vals.index.names) == 2
            assert (
                len(set(outcomes_df.index.names).intersection({"genc_id", "hours_in"}))
                == 2
            )
            assert (
                len(set(outcomes_df.index.names).intersection({"genc_id", "hours_in"}))
                == 2
            )

        print(outcomes_df.head())
        print(mean_vals.head())

        outcomes_df = outcomes_df.replace("", np.nan)

        # print(outcomes_df['patient_id'].isna().sum())
        outcomes_df = outcomes_df.reindex(new_index)
        # now ffill the patient_id and hospital_id for all the new hours in we just added
        # print(outcomes_df['patient_id'].isna().sum())
        outcomes_df[["patient_id", "hospital_id"]] = (
            outcomes_df[["patient_id", "hospital_id"]]
            .groupby("genc_id")[["patient_id", "hospital_id"]]
            .ffill()
            .groupby("genc_id")[["patient_id", "hospital_id"]]
            .bfill()
        )
        # print(outcomes_df['patient_id'].isna().sum())
        outcomes_df = outcomes_df.reset_index().set_index(
            ["patient_id", "genc_id", "hours_in", "hospital_id"]
        )

        mean_vals = mean_vals.reindex(new_index)
        # now ffill the patient_id and hospital_id for all the new hours in we just added
        # print('mean_vals: ',mean_vals['patient_id'].isna().sum())
        mean_vals[["patient_id", "hospital_id"]] = (
            mean_vals[["patient_id", "hospital_id"]]
            .groupby("genc_id")[["patient_id", "hospital_id"]]
            .ffill()
            .groupby("genc_id")[["patient_id", "hospital_id"]]
            .bfill()
        )
        # print(mean_vals['patient_id'].isna().sum())
        mean_vals = mean_vals.reset_index().set_index(
            ["patient_id", "genc_id", "hours_in", "hospital_id"]
        )

        assert all(
            [
                i == 1
                for i in outcomes_df.reset_index()
                .groupby("genc_id")["patient_id"]
                .nunique()
                .values
            ]
        )

        # assert that there are no missing patient ids or hospital ids
        # assert np.isnan(outcomes_df.index.get_level_values('patient_id')).sum()==0
        assert np.isnan(outcomes_df.index.get_level_values("hospital_id")).sum() == 0
        # assert np.isnan(mean_vals.index.get_level_values('patient_id')).sum()==0
        assert np.isnan(mean_vals.index.get_level_values("hospital_id")).sum() == 0

        # forward fill outcomes_df
        idx = pd.IndexSlice
        min_index = (
            outcomes_df.reset_index("hours_in")
            .groupby(["patient_id", "genc_id", "hospital_id"])
            .min()
            .set_index("hours_in", append=True)
            .swaplevel(i="hours_in", j="hospital_id")
            .index
        )  # back to patient_id, genc_id, hours_in, hospital_id
        outcomes_df.loc[min_index, :] = outcomes_df.loc[min_index, :].fillna(
            0
        )  # fill all the minimum values with 0
        # first the resuscitation columns must be forward filled witha  limit

        # currently aggregation window isn't applied yet.
        print(outcomes_df.head())
        outcomes_df["resus_24"] = outcomes_df["resus_24"].ffill(limit=24).fillna(0)
        outcomes_df["resus_48"] = outcomes_df["resus_24"].ffill(limit=48).fillna(0)
        # ffill the rest of the outcomes.
        outcomes_df = outcomes_df.ffill()

        # do we want to add a gap time at the end for outcomes to prevent leakage?

        # formerly the groups were already hourly, but now we need to aggregate.
        # mean_vals.index=mean_vals.index.set_levels(mean_vals.index.levels[2]*args.aggregation_window, level=2)
        # outcomes_df.index=outcomes_df.index.set_levels(outcomes_df.index.levels[2]*args.aggregation_window, level=2)

        # time at zero is a groupby mean
        mean_vals_0 = (
            mean_vals.loc[mean_vals.index.get_level_values("hours_in") < 0, :]
            .groupby(["patient_id", "genc_id", "hospital_id"])
            .mean()
        )

        mean_vals_0["hours_in"] = 0
        mean_vals_0 = mean_vals_0.reset_index().set_index(
            ["patient_id", "genc_id", "hours_in", "hospital_id"]
        )

        # now make sure mean_vals is 0 and up:
        mean_vals = mean_vals.loc[mean_vals.index.get_level_values("hours_in") >= 0, :]

        print(mean_vals_0.head())
        # create new column which is hours in aggregator
        mean_vals["agg_hours_in"] = (
            mean_vals.index.get_level_values("hours_in")
            // int(config.aggregation_window)
            + 1
        ) * int(config.aggregation_window)
        mean_vals = mean_vals.groupby(
            ["patient_id", "genc_id", "agg_hours_in", "hospital_id"]
        ).mean()
        mean_vals.index.names = ["patient_id", "genc_id", "hours_in", "hospital_id"]
        mean_vals = mean_vals.append(mean_vals_0).sort_index()

        # time at zero is a groupby mean
        outcomes_df_0 = (
            outcomes_df.loc[outcomes_df.index.get_level_values("hours_in") < 0, :]
            .groupby(["patient_id", "genc_id", "hospital_id"])
            .max()
        )

        outcomes_df_0["hours_in"] = 0
        outcomes_df_0 = outcomes_df_0.reset_index().set_index(
            ["patient_id", "genc_id", "hours_in", "hospital_id"]
        )

        # now make sure mean_vals is 0 and up:
        outcomes_df = outcomes_df.loc[
            outcomes_df.index.get_level_values("hours_in") >= 0, :
        ]

        # create new column which is hours in aggregator
        outcomes_df["agg_hours_in"] = (
            outcomes_df.index.get_level_values("hours_in")
            // int(config.aggregation_window)
            + 1
        ) * int(config.aggregation_window)
        outcomes_df = outcomes_df.groupby(
            ["patient_id", "genc_id", "agg_hours_in", "hospital_id"]
        ).max()
        outcomes_df.index.names = ["patient_id", "genc_id", "hours_in", "hospital_id"]
        outcomes_df = outcomes_df.append(outcomes_df_0).sort_index()

        # TODO add gap times
        # outcomes_df = add_gap_time(outcomes_df)

        final_data = simple_imput(mean_vals)

        # write out dataframes to hdf
        print("Saving data to disk: " + config.output)
        dynamic_hd5_filt_filename = "all_hourly_data.h5"
        outcomes_df.to_hdf(
            os.path.join(config.output, dynamic_hd5_filt_filename),
            f"interventions_{site}",
        )
        final_data.to_hdf(
            os.path.join(config.output, dynamic_hd5_filt_filename),
            f"vitals_labs_{site}",
        )

    return data


def labs(args, engine):
    idx = pd.IndexSlice
    query_1 = "SELECT REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ') as unique_lab_names, COUNT(REPLACE(REPLACE(LOWER(lab.lab_test_name_raw),'.','') , '-', ' ')) as unique_lab_counts FROM lab GROUP BY unique_lab_names ORDER BY unique_lab_counts ASC;"  # select all lab.test_name_raw (equivalent to itemids in mimiciii)

    df = pd.read_sql(query_1, con=engine)

    # we lose aboutr 20k observations
    unique_items = df.loc[df["unique_lab_counts"] >= 100, "unique_lab_names"].values

    # min percent
    # get the count of unique participants
    if args.min_percent >= 1:
        args.min_percent = args.min_percent / 100

    # get the hospitals in the dataframes
    query_2 = "SELECT DISTINCT hospital_id FROM ip_administrative;"

    dataset_hospitals = pd.read_sql(query_2, con=engine)
    return_dfs = {}

    for site in dataset_hospitals.values.ravel():
        print(site)
        assert (
            site in HOSPITAL_ID.keys()
        ), f"Could not find site {site} in constants.py HOSPITAL_ID dict"
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

        with open(f"{site}_lab_query3.sql", "w") as f:
            f.write(query)

        do_load = True  # set to false to reload the query (takes ~ 20 minutes for 3000 patients)
        if (
            os.path.exists(os.path.join(args.output, f"{site}_tmp_labs2.csv"))
            and do_load
        ):
            print("loading from disk")
            df = pd.read_csv(os.path.join(args.output, f"{site}_tmp_labs2.csv"))
            df.set_index(["patient_id", "genc_id", "hours_in"], inplace=True)
        else:
            print("constructing from scratch")
            time_old = time.time()
            df = pd.read_sql(query, con=engine)
            print(time.time() - time_old)
            # drop the duplicated genc_id column
            df = df.loc[:, ~df.columns.duplicated()]

            df["hours_in"] = (df["hours_in"].values / args.aggregation_window).astype(
                int
            ) * int(args.aggregation_window)
            df.set_index(["patient_id", "genc_id", "hours_in"], inplace=True)

            # Unstack columns
            print("successfully queried, now unstacking columns")

            # replace the drug_screen values with the True/False.

            print("TODO: convert DRUG SCREEN to categorical yes/no")
            print(
                set(
                    df.loc[
                        df["test_name_raw_clean"].isin(DRUG_SCREEN),
                        "result_value_clean",
                    ].values.tolist()
                )
            )

            print(
                df.loc[
                    df["test_name_raw_clean"].isin(DRUG_SCREEN),
                    ["result_value_clean", "test_name_raw_clean"],
                ]
                .groupby("result_value_clean")
                .count()
            )

            print(df.columns.tolist())

            df.loc[~df["result_value_clean"].isna(), "result_value_clean"] = df.loc[
                ~df["result_value_clean"].isna(), "result_value_clean"
            ].apply(x_to_numeric)

            df["result_value_clean"] = pd.to_numeric(df["result_value_clean"], "coerce")

            # sort out all of the drug screen cols to be True/False
            for col in tqdm(DRUG_SCREEN, desc="drug_screen"):
                df.loc[
                    (df["test_name_raw_clean"].isin(DRUG_SCREEN))
                    & (
                        (~df["result_unit_clean"].isna())
                        & (~df["result_value_clean"].isna())
                    )
                    & (df["result_value_clean"] > 0),
                    "result_value_clean",
                ] = 1.0
                # make the unit NaN
                df.loc[
                    (df["test_name_raw_clean"].isin(DRUG_SCREEN))
                    & (~df["result_unit_clean"].isna())
                    & (~df["result_value_clean"].isna()),
                    "result_unit_clean",
                ] = np.nan

            print(df.columns.tolist())
            df.reset_index().to_csv(os.path.join(args.output, f"{site}_tmp_labs2.csv"))

        # --------------------------------------------    UNIT CONVERSION -------------------------------------------------------------------

        units_dict = (
            df[["hospital_id", "test_name_raw_clean", "result_unit_clean"]]
            .groupby(["hospital_id", "test_name_raw_clean"])["result_unit_clean"]
            .apply(name_count)
            .to_dict()
        )

        conversion_list = convert_units(units_dict)

        df_cols = df.columns.tolist()

        # apply the scaling we found to the df
        for item in tqdm(conversion_list, desc="unit_conversion"):
            hospital_id, lab_name, from_unit, scale, to_unit, count = item
            if clean(lab_name)[1] not in df_cols:
                print(clean(lab_name))
                continue
            assert lab_name == clean(
                lab_name
            ), f"{lab_name} is not the same as {clean(lab_name)}"
            df.loc[
                (df["hospital_id"] == hospital_id)
                & (df["test_name_raw_clean"] == lab_name)
                & (df["result_unit_clean"] == from_unit),
                clean(lab_name)[1],
            ] *= scale
            df.loc[
                (df["hospital_id"] == hospital_id)
                & (df["test_name_raw_clean"] == lab_name)
                & (df["result_unit_clean"] == from_unit),
                "result_unit_clean",
            ] = to_unit

        # --------------------------------------------    UNIT CONVERSION FOR UNKNOWN UNITS -------------------------------------------------------------------

        units_dict = (
            df[["hospital_id", "test_name_raw_clean", "result_unit_clean"]]
            .groupby(["hospital_id", "test_name_raw_clean"])["result_unit_clean"]
            .apply(name_count)
            .to_dict()
        )

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
                    comparison = df.loc[
                        (df["hospital_id"] == k[0])
                        & (df["test_name_raw_clean"] == k[1])
                        & (df["result_unit_clean"] == v[0][0]),
                        "result_value_clean",
                    ].values
                elif np.isnan(v[0][0]):
                    # comparison = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean'].isna()), clean(k[1])[1]].values
                    comparison = df.loc[
                        (df["hospital_id"] == k[0])
                        & (df["test_name_raw_clean"] == k[1])
                        & (df["result_unit_clean"].isna()),
                        "result_value_clean",
                    ].values
                else:
                    raise
                norm_p = scipy.stats.kstest(comparison, "norm")[1]

                for item in v[1:]:
                    if item[1] < 20:
                        continue
                    if isinstance(item[0], str):
                        # test_unit = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean']==item[0]), clean(k[1])[1]].values
                        test_unit = df.loc[
                            (df["hospital_id"] == k[0])
                            & (df["test_name_raw_clean"] == k[1])
                            & (df["result_unit_clean"] == item[0]),
                            "result_value_clean",
                        ].values
                    elif np.isnan(item[0]):
                        # test_unit = df.loc[(df['hospital_id']==k[0])&(df['test_name_raw_clean']==k[1])&(df['result_unit_clean'].isna()), clean(k[1])[1]].values
                        test_unit = df.loc[
                            (df["hospital_id"] == k[0])
                            & (df["test_name_raw_clean"] == k[1])
                            & (df["result_unit_clean"].isna()),
                            "result_value_clean",
                        ].values
                    else:
                        raise
                    same_test = scipy.stats.mannwhitneyu(comparison, test_unit)[1]
                    print(k[0], k[1], v[0][0], item[0], same_test)

                    log += f"Are {k[0]} and {k[1]} the same? units are {v[0][0]} and {item[0]}, respectively (p={same_test}) \r\n"

        # put log to txt
        with open(os.path.join(args.output, site + "_units_log.txt"), "w") as f:
            f.write(log)

        # print(determined_same)
        if len(determined_same) > 0:
            # print(print(sorted(df.columns.tolist())
            for item in tqdm(determined_same):
                print(item)
                (
                    hospital_id,
                    lab_name,
                    to_unit,
                    from_unit,
                    num_saved,
                    norm_p,
                    same_test,
                ) = item
                if clean(lab_name)[1] not in df_cols:
                    print(clean(lab_name))
                    continue
                if isinstance(from_unit, str):
                    df.loc[
                        (df["hospital_id"] == hospital_id)
                        & (df["test_name_raw_clean"] == lab_name)
                        & (df["result_unit_clean"] == from_unit),
                        "result_unit_clean",
                    ] = to_unit
                else:
                    df.loc[
                        (df["hospital_id"] == hospital_id)
                        & (df["test_name_raw_clean"] == lab_name)
                        & (df["result_unit_clean"].isna()),
                        "result_unit_clean",
                    ] = to_unit

            np.sum(list(zip(*determined_same))[4])

        units_dict = (
            df[["hospital_id", "test_name_raw_clean", "result_unit_clean"]]
            .groupby(["hospital_id", "test_name_raw_clean"])["result_unit_clean"]
            .apply(name_count)
            .to_dict()
        )

        # eliminate features with unmatchable units
        keep_vars = []
        for k, v in units_dict.items():
            keep_vars.append((k[0], k[1], v[0][0]))

        print(len(df))
        print(
            sum(
                pd.Series(
                    list(
                        zip(
                            df["hospital_id"],
                            df["test_name_raw_clean"],
                            df["result_unit_clean"],
                        )
                    )
                ).isin(keep_vars)
            )
            / len(df)
        )
        index = (
            pd.Series(
                list(
                    zip(
                        df["hospital_id"],
                        df["test_name_raw_clean"],
                        df["result_unit_clean"],
                    )
                )
            )
            .isin(keep_vars)
            .values
        )

        df = df.loc[index]

        # drop the text fields prior to the groupby
        # if 'result_unit' in df.columns.tolist():
        #     df.drop(['result_unit', 'result_unit_clean', 'test_name_raw_clean'], axis=1, inplace=True)

        assert "hospital_id" in df.columns

        # sometimes there are multiple hospitals. Let us just make sure that the first hospital is there for all of the patient's encounters
        df = df.drop(labels="hospital_id", axis=1).join(
            df.groupby(["patient_id", "genc_id"])["hospital_id"].first(),
            how="left",
            on=["patient_id", "genc_id"],
        )

        df["hospital_id"] = df["hospital_id"].replace(HOSPITAL_ID)

        df.reset_index("hours_in", inplace=True)
        df["hours_in"] = df["hours_in"].apply(lambda x: min(0, x))
        df.set_index("hours_in", append=True, inplace=True)
        print("doing massive groupby")
        mean_vals = df.groupby(
            [
                "patient_id",
                "genc_id",
                "hours_in",
                "hospital_id",
                "test_name_raw_clean",
                "result_unit_clean",
            ]
        ).mean()  # could also do .agg('mean', 'median', 'std', 'min', 'max') for example.

        print("doing massive unstack")

        mean_vals = mean_vals.unstack(
            level=["test_name_raw_clean", "result_unit_clean"]
        )

        # potentially drop oth level of multiindex
        mean_vals = mean_vals.droplevel(level=0, axis="columns")

        assert "hospital_id" in mean_vals.index.names

        mean_vals = mean_vals.groupby(
            by=mean_vals.columns, axis=1
        ).mean()  # because we made many of the columns the same

        # Now eliminate rows with fewer than args.min_percent data
        drop_cols = []
        mean_vals = mean_vals.dropna(axis="columns", how="all")

        for col in mean_vals.columns:
            print("obs_rate for ", col, (1 - mean_vals[col].isna().mean()))
            if (1 - mean_vals[col].isna().mean()) < args.min_percent:
                print(col, 1 - mean_vals[col].isna().mean())
                # drop the column.
                drop_cols.append(col)

        mean_vals = mean_vals.drop(drop_cols, axis=1)
        mean_vals.sort_index(inplace=True)

        return_dfs.update({site: mean_vals})

    return return_dfs


def outcomes(args, engine):
    """
        discharge_disposition:
        1: Transferred to acute care inpatient institution
        2: transferred to continuing care.
        3: transferred to other
        4: discharged to home or a home setting with support services
        5: discharged home with no support services from an external agency required
        6: signed out
        7: died
        8: cadaveric donor admitted for organ/tissue removal
        9: stillbirth
        10: transfer to another hospital
        12: patient who does not return from a pass
        20: transfer to another ED
        30 transfer to residential care # Transfer to long-term care home (24-hour nursing), mental health and/or addiction treatment centre
    or hospice/palliative care facility
        40: transfer to group/supportive living #Transfer to assisted living/supportive housing or transitional housing, including shelters; these
    settings do not have 24-hour nursing care.
        61: absent without leave AWOL
        62: AMA
        65:did not return from pass/leave
        66: died while on pass leave
        67: suicide out of facility
        72: died in facility
        73: MAID
        74: suicide
        90: transfer to correctional



        We group these as:
        discharge: 4, 5, 30, 40, 90
        mortality: 7, 66?, 72?, 73? # todo ask about 72 (inpatient mortality vs in hosp mortality?)
        acute: 1
        transfer:  2, 3, 10, 20
        suicide: 67, 74
        Leave AMA: 6, 12, 61, 62, 65
        ignored: 8, 9
        remaining: 66, 73
    """
    query = """ SELECT i.patient_id, i.genc_id, i.hospital_id, h.hours_in, h.mort_24, h.mort_48, h.disch_24, h.disch_48, h.resus_24, h.resus_48, h.acute_24, h.acute_48, h.trans_24, h.trans_48, h.suicide_24, h.suicide_48, h.ama_24, h.ama_48 from ((select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -24 AS hours_in,
        1 AS mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition IN (7, 66, 72, 73))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -48 AS hours_in,
        null::int4 as mort_24,
        1 AS mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition IN (7, 66, 72, 73))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -24 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        1 AS disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (4,5,30, 40, 90))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -48 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        1 AS disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (4,5,30, 40, 90))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', g.intervention_episode_start_date - d.admit_date_time)*24 -24 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        1 AS resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
        LEFT OUTER JOIN (select intervention.genc_id,
            intervention.intervention_code,
            intervention.intervention_episode_start_date
            FROM intervention
            WHERE intervention.intervention_episode_start_date is not null) g
        ON d.genc_id = g.genc_id
    WHERE g.intervention_code = '1HZ30JN' OR g.intervention_code = '1GZ30JH' OR g.intervention_code like '1HZ34__')

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', g.intervention_episode_start_date - d.admit_date_time)*24 -48 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        1 AS resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
        LEFT OUTER JOIN (select intervention.genc_id,
            intervention.intervention_code,
            intervention.intervention_episode_start_date
            FROM intervention
            WHERE intervention.intervention_episode_start_date is not null) g
        ON d.genc_id = g.genc_id
    WHERE g.intervention_code = '1HZ30JN' OR g.intervention_code = '1GZ30JH' OR g.intervention_code like '1HZ34__')

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -24 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        1 AS acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition = 1)

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -48 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        1 AS  acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition = 1)

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -24 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as  acute_48,
        1 AS trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (2, 3, 10, 20))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -48 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        1 AS trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (2, 3, 10, 20))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -24 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        1 AS suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (67, 74))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -48 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        1 AS suicide_48,
        null::int4 as ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (67, 74))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -24 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        1 AS ama_24,
        null::int4 as ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (6, 12, 61, 62, 65))

    UNION

    (select distinct
        d.hospital_id,
        d.patient_id_hashed as patient_id,
        d.genc_id,
        DATE_PART('day', d.discharge_date_time - d.admit_date_time)*24 + DATE_PART('hour', d.discharge_date_time - d.admit_date_time) -48 AS hours_in,
        null::int4 as mort_24,
        null::int4 as mort_48,
        null::int4 as disch_24,
        null::int4 as disch_48,
        null::int4 as resus_24,
        null::int4 as resus_48,
        null::int4 as acute_24,
        null::int4 as acute_48,
        null::int4 as trans_24,
        null::int4 as trans_48,
        null::int4 as suicide_24,
        null::int4 as suicide_48,
        null::int4 as ama_24,
        1 AS ama_48
    FROM ip_administrative d
    WHERE d.discharge_disposition  IN (6, 12, 61, 62, 65))) h

    LEFT JOIN (SELECT ip_administrative.patient_id_hashed AS patient_id,
          ip_administrative.genc_id,
          ip_administrative.hospital_id
          FROM ip_administrative) i
      ON h.genc_id=i.genc_id

    ORDER BY patient_id, genc_id ASC, hours_in ASC;
    """
    outcomes_df = pd.read_sql(query, con=engine)

    assert set(outcomes_df["hospital_id"].values).issubset(
        set(list(HOSPITAL_ID.keys()) + list(HOSPITAL_ID.values()) + ["", np.nan, None])
    )

    outcomes_df["hours_in"] = (
        outcomes_df["hours_in"].values / args.aggregation_window
    ).astype(int)
    outcomes_df.set_index(["patient_id", "genc_id", "hours_in"], inplace=True)

    # outcomes happening before 0 hours, are set to happen at 0 hours
    hours_index = np.asarray(list(set(outcomes_df.index.get_level_values("hours_in"))))
    hours_index = hours_index[hours_index <= 0]

    idx = pd.IndexSlice

    zero_values = (
        outcomes_df.loc[idx[:, :, hours_index], :]
        .groupby(["patient_id", "genc_id"])
        .max()
    )

    # do the groupby with the aximum outcome (i.e. 1 or 0)
    outcomes_df = outcomes_df.groupby(["patient_id", "genc_id", "hours_in"]).max()

    outcomes_df.loc[idx[:, :, 0], :] = zero_values

    # now eliminate all of the negative indices
    hours_index = np.asarray(list(set(outcomes_df.index.get_level_values("hours_in"))))
    hours_index = hours_index[hours_index >= 0]

    outcomes_df = outcomes_df.loc[idx[:, :, hours_index], :]

    outcomes_df.sort_index(inplace=True)

    return outcomes_df


def binary_legth_of_stay(l):
    return 1 if l >= 7 else 0  # TODO: replace with parameter


# add a column to signal training/val or test
def split(data, config):
    #     Create the train and test folds: default test set is 2015. All patients in 2015 will be not be used for training
    #     or validation. Default validation year is 2014. All patients in the validation year will not be used for training.
    # TODO: implement configurable train set - getting all except val/test in train set right now.

    #
    # set a new column for use_train_val
    data["train"] = 1
    data["test"] = 0
    data["val"] = 0
    if config.split_column in ("year", "hospital_id"):
        test_const = int(config.test_split)
        val_const = int(config.val_split)
    else:
        test_const = config.test_split
        val_const = config.val_split

    data.loc[data[config.split_column] == test_const, "train"] = 0
    data.loc[data[config.split_column] == test_const, "test"] = 1
    data.loc[data[config.split_column] == test_const, "val"] = 0
    data.loc[data[config.split_column] == val_const, "train"] = 0
    data.loc[data[config.split_column] == val_const, "test"] = 0
    data.loc[data[config.split_column] == val_const, "val"] = 1
    # check for overlapping patients in test and train/val sets
    if not (
        set(
            data.loc[data[config.split_column] == test_const, "patient_id"].values
        ).isdisjoint(
            set(data.loc[data[config.split_column] != test_const, "patient_id"])
        )
    ):
        # remove patients
        s = sum(data["train"].values)
        patients = set(
            data.loc[data[config.split_column] == test_const, "patient_id"]
        ).intersection(
            set(data.loc[data[config.split_column] != test_const, "patient_id"])
        )
        # print('size {:d}'.format(len(patients)))
        # print(data.loc[data['patient_id'].isin(list(patients))&data['train']==1].shape[0])
        data.loc[
            (data["patient_id"].isin(list(patients)))
            & (data[config.split_column] != test_const),
            "train",
        ] = 0
        data.loc[
            (data["patient_id"].isin(list(patients)))
            & (data[config.split_column] != test_const),
            "val",
        ] = 0
        print(
            "Removed {:d} entries from the training and validation sets because the patients appeared in the test set".format(
                s - sum(data["train"].values)
            )
        )

    if not (
        set(
            data.loc[data[config.split_column] == val_const, "patient_id"].values
        ).isdisjoint(
            set(data.loc[data[config.split_column] != val_const, "patient_id"])
        )
    ):
        # remove patients
        s = sum(data["train"].values)
        patients = set(
            data.loc[data[config.split_column] == val_const, "patient_id"]
        ).intersection(
            set(data.loc[data[config.split_column] < val_const, "patient_id"])
        )
        data.loc[
            (data["patient_id"].isin(list(patients)))
            & (data[config.split_column] != val_const),
            "train",
        ] = 0
        print(
            "Removed {:d} entries from the training set because the patients appeared in the validation set".format(
                s - sum(data["train"].values)
            )
        )

    train_size = data.loc[data["train"] == 1].shape[0]
    val_size = data.loc[data["val"] == 1].shape[0]
    test_size = data.loc[data["test"] == 1].shape[0]

    print(
        "Train set size = {train}, val set size = {val}, test set size = {test}".format(
            train=train_size, val=val_size, test=test_size
        )
    )
    return data


def transform(data):
    # convert length of stay feature
    # 1 - more then 7 days, 0 - less
    data["los"] = data["los"].apply(binary_legth_of_stay)
    data["hospital_id"] = data["hospital_id"].replace(HOSPITAL_ID)
    data = transform_diagnosis(data)
    return data
