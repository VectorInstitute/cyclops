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
    encounter_missingness_threshold: float, optional
        Remove encounters with greater than a fraction of missing features.
        By default, no encounters are removed.
    feature_missingness_threshold: float, optional
        Remove entire feature if more than specified fraction of
        encounters have missing values for that feature.

    """

    strategy: Optional[str] = None
    encounter_missingness_threshold: Optional[float] = 0.0
    feature_missingness_threshold: Optional[float] = 0.0
        
        
def remove_if_missing(features: pd.DataFrame, encounter_missingness_threshold: float, feature_missingness_threshold: float) -> pd.DataFrame:
    """Remove encounters or features if missingness is above input thresholds.
    
    Parameters
    ----------
    features: pandas.DataFrame
        Input features before removal based on missingness and imputation.
    encounter_missingness_threshold: float
        Remove encounters with greater than a fraction of missing features.
        By default, no encounters are removed.
    feature_missingness_threshold: float
        Remove entire feature if more than specified fraction of
        encounters have missing values for that feature.
    
    Returns
    -------
    pandas.DataFrame
        Features after removal based on missingness.
    
    """
    num_na_rows = features.isna().sum()
    if num_na / len(row) > encounter_missingness_threshold:
            
        

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
    features = remove_if_missing(features, imputer.encounter_missingness_threshold, imputer.feature_missingness_threshold)
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
