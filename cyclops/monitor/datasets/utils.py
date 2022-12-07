"""Utilities for datasets."""

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def synthetic_gemini_dataset(size=1000):
    """Create a synthetic Gemini dataset."""
    gemini_columns = [
        "encounter_id",
        "subject_id",
        "city",
        "province",
        "country",
        "language",
        "total_direct_cost",
        "total_indirect_cost",
        "total_cost",
        "hospital_id",
        "sex",
        "age",
        "admit_timestamp",
        "discharge_timestamp",
        "admit_category",
        "discharge_disposition",
        "responsibility_for_payment",
        "province_territory_issuing_health_card_number",
        "number_of_alc_days",
        "institution_from",
        "institution_from_type",
        "institution_to",
        "institution_to_type",
        "readmission",
        "residence_code",
        "admitting_service_raw",
        "discharging_service_raw",
        "mrp_service",
        "cmg",
        "admitting_physician",
        "discharging_physician",
        "mrp",
        "admitting_physician_gim",
        "discharging_physician_gim",
        "mrp_gim",
        "admitting_service_mapped",
        "discharging_service_mapped",
        "from_nursing_home_mapped",
        "from_acute_care_institution_mapped",
        "los_derived",
        "del_present",
        "gemini_cohort",
        "discharge_description",
        "admit_via_ambulance",
        "triage_level",
        "physician_initial_assessment_date_time",
        "er_admit_timestamp",
        "disposition_date_time",
        "er_discharge_timestamp",
        "length_of_stay_in_er",
        "mortality",
    ]

    # get dataframe mapping from gemini.yaml
    gemini_mapping = OmegaConf.load(
        "cyclops/monitor/datasets/configs/gemini.yaml"
    ).metadata_mapping

    df = pd.DataFrame(columns=gemini_columns)

    df["encounter_id"] = np.random.randint(0, 100000, size=size)
    df["subject_id"] = np.random.randint(0, 100000, size=size)
    df["city"] = np.random.choice(["Toronto", "Ottawa", "Montreal"], size=size)
    df["province"] = np.random.choice(["Ontario", "Quebec", "Alberta"], size=size)
    df["country"] = np.random.choice(["Canada", "USA", "Mexico"], size=size)
    df["language"] = np.random.choice(["English", "French", "Spanish"], size=size)
    df["total_direct_cost"] = np.random.randint(0, 100000, size=size)
    df["total_indirect_cost"] = np.random.randint(0, 100000, size=size)
    df["total_cost"] = np.random.randint(0, 100000, size=size)
    df["hospital_id"] = np.random.choice(gemini_mapping["hospital_type"], size=size)
    df["sex"] = np.random.choice(["M", "F"], size=size)
    df["age"] = np.random.randint(0, 100, size=size)
    df["admit_timestamp"] = pd.date_range(
        start="1/1/2015", end="8/1/2020", periods=size
    )
    df["discharge_timestamp"] = pd.date_range(
        start="1/1/2015", end="8/1/2020", periods=size
    )
    df["mortality"] = np.random.randint(0, 2, size=size)

    X = np.random.rand(size, 64, 7)

    return df, X, gemini_mapping
