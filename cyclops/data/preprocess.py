"""Preprocessing scripts for converting misc.

datasets to huggingface datasets.

"""
import os

import pandas as pd


def nihcxr_preprocess(
    df: pd.DataFrame, nihcxr_dir: str, image_key: str = "features"
) -> pd.DataFrame:
    """Preprocess NIHCXR dataframe.

    Add a column with the path to the image and create
    one-hot encoded pathogies from Finding Labels column.

    Parameters
    ----------
        df (pd.DataFrame): NIHCXR dataframe.

    Returns
    -------
        pd.DataFrame: pre-processed NIHCXR dataframe.

    """
    # Add path column
    df[image_key] = df["Image Index"].apply(
        lambda x: os.path.join(nihcxr_dir, "images", x)
    )

    # Create one-hot encoded pathologies
    pathologies = df["Finding Labels"].str.get_dummies(sep="|")

    # Add one-hot encoded pathologies to dataframe
    df = pd.concat([df, pathologies], axis=1)

    return df
