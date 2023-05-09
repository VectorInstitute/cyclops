"""Convenient functions for loading datasets as Huggingface datasets."""

import os

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets.features import Image

from cyclops.data.preprocess import nihcxr_preprocess


def load_nihcxr(path: str) -> Dataset:
    """Load NIH Chest X-Ray dataset as a Huggingface dataset."""
    df = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"))
    df = nihcxr_preprocess(df, path)
    nih_ds = Dataset.from_pandas(df, preserve_index=False)
    nih_ds = nih_ds.add_column(
        "timestamp",
        pd.date_range(start="1/1/2019", end="12/25/2019", periods=nih_ds.num_rows),
    )
    nih_ds = nih_ds.cast_column("features", Image(decode=True))
    return nih_ds
