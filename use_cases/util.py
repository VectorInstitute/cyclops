"""Utility functions shared across use-cases."""

import importlib
import types
from functools import reduce
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from cyclops.data.utils import is_out_of_core
from cyclops.process.column_names import EVENT_NAME, EVENT_VALUE
from cyclops.utils.file import yield_dataframes


def get_use_case_params(dataset: str, use_case: str) -> types.ModuleType:
    """Import parameters specific to each use-case.

    Parameters
    ----------
    dataset: str
        Name of the dataset, e.g. mimiciv.
    use_case: str
        Name of the use-case, e.g. mortality_decompensation.

    Returns
    -------
    types.ModuleType
        Imported constants module with use-case parameters.

    """
    return importlib.import_module(
        ".".join(["use_cases", "params", dataset, use_case, "constants"])
    )


def get_top_events(events_path: str, n_events: int) -> np.ndarray:
    """Get top events from events data saved in batches.

    Parameters
    ----------
    events_path : str
        Path to the directory of saved events.
    n_events : int
        The number of top events.

    Returns
    -------
    np.ndarray
        The array of the top events names.

    """
    all_top_events = []
    for _, events in enumerate(yield_dataframes(events_path, log=False)):
        top_events = (
            events[EVENT_NAME][~events[EVENT_VALUE].isna()]
            .value_counts()[:n_events]
            .index
        )

        all_top_events.append(top_events)

        del events

    # Take only the events common to every file
    top_events = reduce(np.intersect1d, tuple(all_top_events))
    return sorted(top_events)


def valid_events(events: pd.DataFrame, top_events: np.ndarray) -> pd.DataFrame:
    """Keep the events that are included in the top events.

    Parameters
    ----------
    events : pd.DataFrame
        The events dataframe.
    top_events : np.ndarray
        The list of top events.

    Returns
    -------
    pd.DataFrame
        The events dataframe including only top events.

    """
    return events[events[EVENT_NAME].isin(top_events)]


def get_pandas_df(
    dataset: Union[Dataset, DatasetDict, Mapping],
    feature_cols: Optional[List[str]] = None,
    label_cols: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, pd.Series], Dict[str, Tuple[pd.DataFrame, pd.Series]]]:
    """Convert dataset to pandas dataframe.

    NOTE: converting to pandas does not work with IterableDataset/IterableDatasetDict
    (i.e. when dataset is loaded with stream=True). So, this function should only be
    used with datasets that are loaded with stream=False and are small enough to fit
    in memory. Use :func:`is_out_of_core` to check if dataset is too large to fit in
    memory.


    Parameters
    ----------
    dataset : Union[Dataset, DatasetDict, Mapping]
        Dataset to convert to pandas dataframe.
    feature_cols : List[str], optional
        List of feature columns to include in the dataframe, by default None
    label_cols : str, optional
        Label column to include in the dataframe, by default None

    Returns
    -------
    Union[Tuple[pd.DataFrame, pd.Series], Dict[str, Tuple[pd.DataFrame, pd.Series]]]
        Pandas dataframe or dictionary of pandas dataframes.

    Raises
    ------
    TypeError
        If dataset is not a Dataset, DatasetDict, or Mapping.

    """
    if isinstance(dataset, (DatasetDict, Mapping)):
        return {
            k: get_pandas_df(v, feature_cols=feature_cols, label_cols=label_cols)
            for k, v in dataset.items()
        }
    if isinstance(dataset, Dataset) and not is_out_of_core(dataset.dataset_size):
        # validate feature_cols and label_col
        if feature_cols is not None and not set(feature_cols).issubset(
            dataset.column_names
        ):
            raise ValueError("feature_cols must be a subset of dataset column names.")
        if label_cols is not None and not set(label_cols).issubset(
            dataset.column_names
        ):
            raise ValueError("label_col must be a column name of dataset.")

        df = dataset.to_pandas(batched=False)  # set batched=True for large datasets

        if feature_cols is not None and label_cols is not None:
            pd_dataset = (df[feature_cols], df[label_cols])
        elif label_cols is not None:
            pd_dataset = (df.drop(label_cols, axis=1), df[label_cols])
        elif feature_cols is not None:
            pd_dataset = (df[feature_cols], None)
        else:
            pd_dataset = (df, None)
        return pd_dataset

    raise TypeError(
        f"Expected dataset to be a Dataset or DatasetDict. Got: {type(dataset)}"
    )
