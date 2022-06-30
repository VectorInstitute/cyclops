"""Imputation functions."""

import logging
from dataclasses import dataclass

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.processors.column_names import ENCOUNTER_ID
from cyclops.processors.constants import MEAN, MEDIAN
from cyclops.processors.util import is_timeseries_data
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
    imputefunc: str
    encounter_missingness_threshold: float, optional
        Remove encounters with greater than a fraction of missing features.
        By default, no encounters are removed.
    feature_missingness_threshold: float, optional
        Remove entire feature if more than specified fraction of
        encounters have missing values for that feature. By default, no features
        are removed.

    """

    encounter_missingness_threshold: float = 0.0
    feature_missingness_threshold: float = 0.0
    imputefunc: str = MEAN

    def remove_encounters_if_missing(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove encounters if missingness is above certain threshold.

        For static, missingness is no. of missing features / total no. of features.
        For temporal features, missingness is no. of missing features summed
        over timesteps / (total no. of features * timesteps).

        Parameters
        ----------
        features: pandas.DataFrame
            Input features before removal based on missingness and imputation.

        Returns
        -------
        pandas.DataFrame
            Features after removal based on missingness.

        """
        # If time-series features, then missingness computed over timesteps as well.
        if is_timeseries_data(features):
            encounter_ids = set(features.index.get_level_values(0))
            for encounter_id in encounter_ids:
                features_encounter = features.loc[encounter_id]
                num_na_features = features_encounter.isna().sum().sum()
                fraction_missing = num_na_features / (
                    len(features_encounter) * len(features.columns)
                )
                if fraction_missing > self.encounter_missingness_threshold:
                    features = features.drop(encounter_id, level=ENCOUNTER_ID)
            num_encounters_dropped = len(encounter_ids) - len(
                set(features.index.get_level_values(0))
            )
            LOGGER.info(
                "Dropped %d encounters, due to missingness!", num_encounters_dropped
            )

            return features

        encounter_ids = features.index
        fraction_missing = features.isna().sum(axis=1) / len(features.columns)
        features = features[fraction_missing <= self.encounter_missingness_threshold]
        num_encounters_dropped = len(encounter_ids) - len(features.index)
        LOGGER.info(
            "Dropped %d encounters, due to missingness!", num_encounters_dropped
        )

        return features

    def remove_features_if_missing(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove features if missingness is above certain threshold.

        Parameters
        ----------
        features: pandas.DataFrame
            Input features before removal based on missingness and imputation.

        Returns
        -------
        pandas.DataFrame
            Features after removal based on missingness.

        """
        fraction_missing = features.isna().sum(axis=0) / len(features)
        for col in fraction_missing.index:
            if fraction_missing[col] > self.feature_missingness_threshold:
                LOGGER.info(
                    "Dropping %s feature, missingness is higher than threshold!", col
                )
                features = features.drop(columns=[col])

        return features

    @time_function
    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in features.

        Before applying imputation, this function also removes encounters, if
        they have more than a threshold of missing features. Features which have
        missingness above a threshold over all encounters are also removed.

        Parameters
        ----------
        features: pandas.DataFrame
            Input features before imputation.

        Returns
        -------
        pandas.DataFrame
            Features after imputation.

        """
        features = features.copy()
        if self.encounter_missingness_threshold > 0:
            features = self.remove_encounters_if_missing(
                features,
            )
        if self.feature_missingness_threshold > 0:
            features = self.remove_features_if_missing(
                features,
            )

        if self.imputefunc == MEAN:
            per_column_impute_values = features.mean(axis=0, skipna=True)
        if self.imputefunc == MEDIAN:
            per_column_impute_values = features.median(axis=0, skipna=True)

        for col, per_column_impute_value in per_column_impute_values.items():
            features[[col]] = features[[col]].fillna(per_column_impute_value)

        return features
