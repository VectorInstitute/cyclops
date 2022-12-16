"""Utility functions shared across use-cases."""

import importlib
import types
from functools import reduce

import numpy as np
import pandas as pd

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
    return top_events


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
