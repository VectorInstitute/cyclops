"""Convenient functions for loading datasets as Huggingface datasets."""

import os

import pandas as pd
from datasets.arrow_dataset import Dataset
from datasets import DatasetDict
from datasets.features import Image
from datasets.features import Features, Value

from cyclops.data.preprocess import nihcxr_preprocess
from sklearn.model_selection import GroupShuffleSplit
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

def load_nihcxr(path: str, image_column: str = "image", add_timestamps: bool = True, progress = False) -> Dataset:
    """Load NIH Chest X-Ray dataset as a Huggingface dataset."""
    if not progress:
        disable_progress_bar()

    df = pd.read_csv(os.path.join(path, "Data_Entry_2017.csv"))
    df = nihcxr_preprocess(df, path, image_key=image_column)
    # split into train and test using text files
    train_id, test_id = [], []
    with open(os.path.join(path, "train_val_list.txt"), "r") as f:
        for line in f:
            train_id.append(line.strip())
    with open(os.path.join(path, "test_list.txt"), "r") as f:
        for line in f:
            test_id.append(line.strip())

    # create dataset_dict based on train test split
    train_df = df[df["Image Index"].isin(train_id)]
    test_df = df[df["Image Index"].isin(test_id)]

    nih_ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    # cast pathologies columns as float
    pathologies = ['Atelectasis',
                   'Cardiomegaly',
                   'Consolidation',
                   'Edema',
                   'Effusion',
                   'Emphysema',
                   'Fibrosis',
                   'Hernia',
                   'Infiltration',
                   'Mass',
                   'Nodule',
                   'Pleural_Thickening',
                   'Pneumonia',
                   'Pneumothorax']
    for pathology in pathologies:
        nih_ds["train"] = nih_ds["train"].cast_column(pathology, Value("float32"))
        nih_ds["test"] = nih_ds["test"].cast_column(pathology, Value("float32"))

    # add timestamp column
    if add_timestamps:
        nih_ds["train"] = nih_ds["train"].add_column(
            "timestamp",
            pd.date_range(start="1/1/2019", end="10/19/2019", periods=nih_ds['train'].num_rows),
        )
        nih_ds["test"] = nih_ds["test"].add_column(
            "timestamp",
            pd.date_range(start="10/20/2019", end="12/25/2019", periods=nih_ds['test'].num_rows),
        )
    nih_ds = nih_ds.cast_column(image_column, Image(decode=True))
    return nih_ds