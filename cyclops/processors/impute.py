"""Imputation functions."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.constants import MEAN, MEDIAN
from cyclops.utils.log import setup_logging
from cyclops.utils.profile import time_function

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


@dataclass
class Imputer:
    """Imputation options.

    Parameters
    ----------
    strategy: str, optional
        Imputation strategy for features.

    """

    strategy: Optional[str] = None


@time_function
def impute_features(features: pd.DataFrame, imputer: Imputer) -> pd.DataFrame:
    """Impute missing values in features.

    Parameters
    ----------
    features: pandas.DataFrame
        Input features before imputation.
    imputer: Imputer
        Imputation options.

    Returns
    -------
    pandas.DataFrame
        Features after imputation.

    """
    if not imputer.strategy:
        return features
    if imputer.strategy == MEAN:
        per_column_impute_values = features.mean(axis=0, skipna=True, numeric_only=True)
    if imputer.strategy == MEDIAN:
        per_column_impute_values = features.median(
            axis=0, skipna=True, numeric_only=True
        )

    for col, per_column_impute_value in per_column_impute_values.items():
        features[[col]] = features[[col]].fillna(per_column_impute_value)

    return features
