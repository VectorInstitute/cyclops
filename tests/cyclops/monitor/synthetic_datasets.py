"""Synthetic Datasets."""

import numpy as np
import pandas as pd
from datasets.arrow_dataset import Dataset


def synthetic_gemini_dataset(size=100):
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
    ]

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
    df["hospital_id"] = np.random.choice(
        ["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW", "PMH", "SBK"],
        size=size,
    )
    df["sex"] = np.random.choice(["M", "F"], size=size)
    df["age"] = np.random.randint(0, 100, size=size)
    df["admit_timestamp"] = pd.date_range(
        start="1/1/2015",
        end="8/1/2020",
        periods=size,
    )
    df["discharge_timestamp"] = pd.date_range(
        start="1/1/2015",
        end="8/1/2020",
        periods=size,
    )
    df["mortality"] = np.random.randint(0, 2, size=size)
    df["features"] = np.random.rand(size, 64, 7).tolist()

    return Dataset.from_pandas(df, preserve_index=False)


def synthetic_nih_dataset(size=8):
    """Create a synthetic NIH dataset."""
    nih_columns = [
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
    ]

    df = pd.DataFrame(columns=nih_columns)
    df["encounter_id"] = np.random.randint(0, 100000, size=size)
    df["subject_id"] = np.random.randint(0, 100000, size=size)
    df["city"] = np.random.choice(["Toronto", "Ottawa", "Montreal"], size=size)
    df["province"] = np.random.choice(["Ontario", "Quebec", "Alberta"], size=size)
    df["country"] = np.random.choice(["Canada", "USA", "Mexico"], size=size)
    df["language"] = np.random.choice(["English", "French", "Spanish"], size=size)
    df["total_direct_cost"] = np.random.randint(0, 100000, size=size)
    df["total_indirect_cost"] = np.random.randint(0, 100000, size=size)
    df["total_cost"] = np.random.randint(0, 100000, size=size)
    df["hospital_id"] = np.random.choice(
        ["SMH", "MSH", "THPC", "THPM", "UHNTG", "UHNTW", "PMH", "SBK"],
        size=size,
    )
    df["sex"] = np.random.choice(["M", "F"], size=size)
    df["age"] = np.random.randint(0, 100, size=size)
    df["admit_timestamp"] = pd.date_range(
        start="1/1/2015",
        end="8/1/2020",
        periods=size,
    )

    df["features"] = np.random.rand(size, 1, 224, 224).tolist()
    return Dataset.from_pandas(df, preserve_index=False)


def synthetic_generic_dataset(size=100, concatenate_features=False):
    """Create a synthetic Gemini dataset."""
    columns = ["timestamp"]

    df = pd.DataFrame(columns=columns)
    df["timestamp"] = pd.date_range(start="1/1/2020", end="12/25/2020", periods=size)
    df["discharge_timestamp"] = pd.date_range(
        start="1/1/2015",
        end="8/1/2020",
        periods=size,
    )
    df["mortality"] = np.random.randint(0, 2, size=size)
    if concatenate_features:
        df["features"] = np.random.rand(size, 10).tolist()
    else:
        for i in range(10):
            df[f"feature_{i}"] = np.random.rand(size)
    return Dataset.from_pandas(df, preserve_index=False)
