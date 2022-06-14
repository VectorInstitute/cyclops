"""Feature handling for automatic feature creation from processed data."""


import abc
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from codebase_ops import get_log_file_path
from cyclops.constants import FEATURES
from cyclops.plotter import plot_histogram, plot_temporal_features
from cyclops.processors.constants import (
    BINARY,
    FEATURE_TYPE,
    GROUP,
    MIN_MAX,
    MISSING_CATEGORY,
    NORMALIZATION_METHOD,
    NUMERIC,
    STANDARD,
    STATIC,
    TEMPORAL,
)
from cyclops.processors.impute import Imputer, impute_features
from cyclops.processors.util import is_timeseries_data
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


def _get_scaler_type(normalization_method: str) -> type:
    """Return a scaling object mapped from a string value.

    Parameters
    ----------
    normalization_method: str
        A string specifying which scaler to return.

    Returns
    -------
    type
        An sklearn.preprocessing scaling object.

    """
    scaler_map = {STANDARD: StandardScaler, MIN_MAX: MinMaxScaler}

    # Raise an exception if the normalization string is not recognized.
    if normalization_method not in scaler_map:
        options = ", ".join(["'" + k + "'" for k in scaler_map])
        raise ValueError(
            f"'{normalization_method}' is invalid, must be in None, {options}"
        )

    return scaler_map[normalization_method]


def _category_to_numeric(
    series: pd.Series, inplace: bool = False, unique: Optional[np.ndarray] = None
) -> pd.Series:
    """Convert categorical column to numerical.

    Takes a series and replaces its the values with the index
    of their value in the array's sorted, unique values.

    Parameters
    ----------
    series: pandas.Series
        Input column of categorical values.
    inplace: bool, optional
        Flag to replace values in-place.
    unique: numpy.ndarray, optional
        Pre-computed unique values.

    Returns
    -------
    pandas.Series
        Converted numerical column of values.

    """
    if unique is None:
        unique = np.unique(series.values)

    map_dict: dict = {}
    for i, unique_val in enumerate(unique):
        map_dict[unique_val] = i

    if inplace:
        series.replace(map_dict, inplace=inplace)
        return series

    return series.replace(map_dict, inplace=inplace)


def _attempt_to_numeric(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Attempt conversion of columns of dataframe to numeric.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Output dataframe, with possibly type converted columns.

    """
    for column in dataframe.columns:
        try:
            dataframe[column] = pd.to_numeric(dataframe[column])
        except (ValueError, TypeError):
            pass

    return dataframe


class FeatureMeta(ABC):
    """Abstract feature class to act as parent for concrete feature classes.

    Attributes
    ----------
    feature_type : str
        The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
    is_target: bool
        Is the feature a target variable, True if yes, else False.

    """

    def __init__(self, feature_type: str, is_target: bool) -> None:
        """Instantiate.

        Parameters
        ----------
        feature_type : str
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        is_target: bool
            Is the feature a target variable, True if yes, else False.

        """
        self.feature_type = feature_type
        self.is_target = is_target

    @abstractmethod
    def parse(self, series: pd.Series) -> pd.Series:
        """Parse feature using meta information.

        Parameters
        ----------
        series: pandas.Series
            Input feature column.

        Returns
        -------
        pandas.Series
            Parsed feature column.

        """
        return series

    def get_feature_type(self) -> str:
        """Get feature type.

        Returns
        -------
        str
            Returns feature type.

        """
        return self.feature_type


class BinaryFeatureMeta(FeatureMeta):
    """A class for handling binary features, i.e., with values 0, 1.

    Any acceptable inputs which can be converted to binary values, such
    as an array with only unique values 'A', 'B' will be converted to
    0, 1.

    Attributes
    ----------
    group: str, optional
        Name of group of feature, incase it belongs to a categorical group.

    """

    def __init__(
        self,
        feature_type: str = BINARY,
        is_target: bool = False,
        group: Optional[str] = None,
    ) -> None:
        """Instantiate.

        Parameters
        ----------
        feature_type : str
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        group: str, optional
            Name of group of feature, incase it belongs to a categorical group.
        is_target: bool
            Is the feature a target variable, True if yes, else False.

        """
        super().__init__(feature_type, is_target)

        # Group is used to track the group of a binary categorical variable.
        self.group = group

    def parse(self, series: pd.Series) -> pd.Series:
        """Parse feature using meta information.

        Parameters
        ----------
        series: pandas.Series
            Input feature column.

        Returns
        -------
        pandas.Series
            Parsed feature column.

        """
        unique = np.unique(series.values)
        if len(unique) != 2:
            raise ValueError(
                "Binary features must have two unique values, e.g., [0, 1], ['A', 'B']."
            )
        # Convert strings to numerical binary values.
        if not np.array_equal(unique, np.array([0, 1])):
            series = _category_to_numeric(series, unique=unique)

        return series.astype(np.uint8)


class NumericFeatureMeta(FeatureMeta):
    """A class for handling numeric features, with normalization functionality.

    Attributes
    ----------
    group: str, optional
        Name of group of feature, incase it belongs to a categorical group.

    """

    def __init__(
        self,
        feature_type: str = NUMERIC,
        is_target: bool = False,
        normalization_method: Optional[str] = STANDARD,
    ):
        """Instantiate.

        Parameters
        ----------
        feature_type : str, optional
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        is_target: bool
            Is the feature a target variable, True if yes, else False.
        normalization_method: str, optional
            Name of normalization method, 'standard', 'min-max' or None.

        """
        super().__init__(feature_type, is_target)
        self.normalization_method = normalization_method
        self.scaler = None

    def parse(self, series: pd.Series) -> pd.Series:
        """Parse feature using meta information.

        Parameters
        ----------
        series: pandas.Series
            Input feature column.

        Returns
        -------
        pandas.Series
            Parsed feature column.

        """
        if self.normalization_method is not None:
            scaler = _get_scaler_type(self.normalization_method)
            self.scaler = scaler().fit(series.values.reshape(-1, 1))

        return series

    def scale(self, series: pd.Series) -> pd.Series:
        """Scale a 1D array based on selected scaling object.

        If the scaler is none, it acts as an identity function.

        Parameters
        ----------
        series: pandas.Series
            Input feature column.

        Returns
        -------
        pandas.Series
            Output feature column with scaled values.

        """
        if self.scaler is None:
            return series

        return pd.Series(
            np.squeeze(self.scaler.transform(series.values.reshape(-1, 1))),
            index=series.index,
        )

    def inverse_scale(self, series: pd.Series) -> pd.Series:
        """Apply Inverse scaling to a 1D array based on selected scaling object.

        If the scaler is none, it acts as an identity function.

        Parameters
        ----------
        series: pandas.Series
            Input feature column.

        Returns
        -------
        pandas.Series
            Output feature column with inverse scaled values.

        """
        if self.scaler is None:
            return series

        return pd.Series(
            np.squeeze(self.scaler.inverse_transform(series.values.reshape(-1, 1))),
            index=series.index,
        )


class FeatureHandler:
    """Feature handler class.

    Attributes
    ----------
    meta: dict
        Dictionary to store meta information for each feature.
    features: pandas.DataFrame
        Container to store features added.
    normalization_method: str
        Normalization method applicable for numeric features.

    """

    def __init__(
        self, features: Optional[pd.DataFrame] = None, normalization_method=STANDARD
    ) -> None:
        """Instantiate.

        Parameters
        ----------
        features: pandas.DataFrame
            Features to add, instantiated along with the feature handler.
        normalization_method: str
            Normalization method applicable for numeric features.

        """
        self.meta: dict = {}
        self.normalization_method = normalization_method
        self.features = {STATIC: pd.DataFrame(), TEMPORAL: pd.DataFrame()}
        self.reference = {STATIC: pd.DataFrame(), TEMPORAL: pd.DataFrame()}
        if features is not None:
            self.add_features(features)

    @property
    def static(self) -> pd.DataFrame:
        """Return static features."""
        return self.features[STATIC]

    @property
    def temporal(self) -> pd.DataFrame:
        """Return temporal features."""
        return self.features[TEMPORAL]

    @property
    def unscaled(self) -> Dict[str, pd.DataFrame]:
        """Return unscaled features."""
        return self.features

    @property
    def scaled(self) -> Dict[str, pd.DataFrame]:
        """Scale and return scaled dataframe."""
        return self._scale()

    @property
    def names(self) -> list:
        """Access as attribute, feature names.

        Returns
        -------
        list
            List of all feature names.

        """
        feature_names = list(self.static.columns) + list(self.temporal.columns)
        return feature_names

    @property
    def types(self) -> dict:
        """Access as attribute, feature types names.

        Note: These are built-in feature names.

        Returns
        -------
        dict
            Feature type mapped for each feature.

        """
        return {f: meta.get_feature_type() for f, meta in self.meta.items()}

    @property
    def targets(self) -> Dict[str, list]:
        """Access as attribute, names of feature columns are potential target variables.

        Returns
        -------
        Dict
            Target feature names (divided into 'static' and 'temporal').

        """
        return {
            STATIC: [col for col in self.static if self.meta[col].is_target],
            TEMPORAL: [col for col in self.temporal if self.meta[col].is_target],
        }

    def _scale(self) -> Dict[str, pd.DataFrame]:
        """Apply scaling."""
        static_features = self.static.copy(deep=True)
        temporal_features = self.temporal.copy(deep=True)

        for col in self.static.columns:
            if self.meta[col].feature_type == NUMERIC:
                static_features[col] = self.meta[col].scale(static_features[col])
        for col in self.temporal.columns:
            if self.meta[col].feature_type == NUMERIC:
                temporal_features[col] = self.meta[col].scale(temporal_features[col])

        return {STATIC: static_features, TEMPORAL: temporal_features}

    def _get_feature_names_type(self, feature_type: str) -> Dict[str, list]:
        """Get names of features added to the handler.

        Returns
        -------
        Dict
            Names of features (divided into 'static' and 'temporal').

        """
        return {
            STATIC: [
                col
                for col in self.static
                if self.meta[col].feature_type == feature_type
            ],
            TEMPORAL: [
                col
                for col in self.temporal
                if self.meta[col].feature_type == feature_type
            ],
        }

    def get_numerical_feature_names(self) -> Dict[str, list]:
        """Get names of all numerical features added to the handler.

        Returns
        -------
        Dict
            Names of numerical features (divided into 'static' and 'temporal').

        """
        return self._get_feature_names_type(NUMERIC)

    def get_categorical_feature_names(self) -> Dict[str, list]:
        """Get names of all categorical features added to the handler.

        Returns
        -------
        Dict
            Names of categorical features (divided into 'static' and 'temporal').

        """
        return self._get_feature_names_type(BINARY)

    def extract_features(self, names: list) -> Dict[str, pd.DataFrame]:
        """Extract features by name.

        Parameters
        ----------
        names: list
            Feature name(s).

        Returns
        -------
        pandas.DataFrame
            Requested features formatted as a DataFrame.

        """
        # Convert to a list if extracting a single feature.
        names = [names] if isinstance(names, str) else names

        # Ensure all features exist.
        inter = set(names).intersection(set(self.names))
        if len(inter) != len(names):
            raise ValueError(f"Features {set(names) - inter} do not exist.")
        static_names = [n for n in names if n in self.static.columns]
        temporal_names = [n for n in names if n in self.temporal.columns]

        return {
            STATIC: self.features[STATIC][static_names],
            TEMPORAL: self.features[TEMPORAL][temporal_names],
        }

    def _add_feature(  # pylint: disable=too-many-arguments
        self,
        series: pd.Series,
        aggregate_type: str,
        feature_meta: abc.ABCMeta,
        init_kwargs: Dict,
        parse_kwargs: Dict,
    ) -> None:
        """Add feature (internal method).

        Parameters
        ----------
        series: pandas.Series
            Feature column to concat.
        feature_meta: abc.ABCMeta
            Feature metadata class.
        init_kwargs: dict
            Init keyword arguments for feature meta.
        parse_kwargs: dict
            Parse keyword arguments for parsing feature using meta information.

        """
        if series.name in self.names:
            LOGGER.warning(
                "Feature %s already exists, possibly rename if its a new feature!",
                series.name,
            )
            return
        meta = feature_meta(**init_kwargs)
        series = meta.parse(series, **parse_kwargs)

        # Add to features.
        self.features[aggregate_type] = pd.concat(
            [self.features[aggregate_type], series], axis=1
        )

        # Add to meta.
        self.meta[series.name] = meta

    def _add_binary(
        self, series: pd.Series, aggregate_type: str, group: Optional[str] = None
    ) -> None:
        """Add binary features.

        Parameters
        ----------
        series: pandas.Series
            Feature column.
        group: str, optional
            Feature type name.

        """
        init_kwargs = {FEATURE_TYPE: BINARY, GROUP: group}
        parse_kwargs: Dict = {}
        self._add_feature(
            series, aggregate_type, BinaryFeatureMeta, init_kwargs, parse_kwargs
        )

    def _add_numerical(self, series: pd.Series, aggregate_type: str) -> None:
        """Add numerical feature.

        Parameters
        ----------
        series: pandas.Series
            Feature column.

        """
        init_kwargs: Dict[str, Any] = {
            FEATURE_TYPE: NUMERIC,
            NORMALIZATION_METHOD: self.normalization_method,
        }
        parse_kwargs: Dict = {}
        self._add_feature(
            series, aggregate_type, NumericFeatureMeta, init_kwargs, parse_kwargs
        )

    def _add_categorical(self, series: pd.Series, aggregate_type: str) -> None:
        """Add categorical feature.

        Parameters
        ----------
        series: pandas.Series
            Feature column.

        """
        unique = np.unique(series)

        binary_names = [
            series.name + "-" + unique_val for unique_val in unique.astype(str)
        ]
        numeric = _category_to_numeric(series, unique=unique)
        onehot = np.zeros((numeric.size, numeric.max() + 1))
        onehot[np.arange(numeric.size), numeric] = 1
        features = pd.DataFrame(onehot, index=series.index, columns=binary_names)

        for col in features:
            self._add_binary(features[col], aggregate_type, group=series.name)

    def set_targets(self, names: Union[str, list, set]) -> None:
        """Set some feature columns to be targets.

        Each feature has meta info, among which is a `is_target` flag
        to track if it may be a target variable.

        Parameters
        ----------
        names: str or list or set
            Names of feature(s) to set as target(s).

        """
        names = set([names]) if isinstance(names, str) else set(names)
        for name in names:
            self.meta[name].is_target = True

    def _instantiate_containers(self, index: pd.Index, aggregate_type: str) -> None:
        """Instantiate feature and reference data containers, with indices.

        Parameters
        ----------
        index: pandas.Index
            Indices to set for feature and reference data containers.
        aggregate_type: str
            'static' or 'temporal', feature containers to instantiate.

        """
        if len(self.features[aggregate_type].index) == 0:
            self.features[aggregate_type] = pd.DataFrame(index=index)
            self.features[aggregate_type].columns.name = FEATURES
            self.reference[aggregate_type] = pd.DataFrame(index=index)

    def add_features(
        self, features: pd.DataFrame, reference_cols: Optional[list] = None
    ) -> None:
        """Add features.

        Parameters
        ----------
        features: pandas.DataFrame
            Features to add.
        reference_cols: list, optional
            Reference columns stored for mapping and creating slices of features
            e.g. (filtering on hospital(s)).

        """
        if not isinstance(features, pd.DataFrame):
            raise ValueError("Input to feature handler must be a pandas.DataFrame.")

        if is_timeseries_data(features):
            aggregate_type = TEMPORAL
        else:
            aggregate_type = STATIC
        self._instantiate_containers(features.index, aggregate_type)

        if reference_cols is None:
            reference_cols = []

        if aggregate_type == STATIC:
            for col in reference_cols:
                self.reference[col] = features[col]
                features.drop(col, axis=1, inplace=True)

        # Attempt to turn any possible columns to numeric.
        features = _attempt_to_numeric(features)

        # Infer other column types.
        features = features.infer_objects()

        for col in features:
            if is_string_dtype(features[col]):
                features[col] = features[col].fillna(MISSING_CATEGORY)
            unique = np.unique(features[col].values)

            # If all values are NaN, feature is dropped.
            if len(unique) == 1 and np.isnan(unique[0]):
                LOGGER.warning("Feature %s has all NaNs, will not be added.", col)
                continue

            # Check if it can be represented as binary.
            # (Numeric or string alike)
            if len(unique) == 2:
                # If one of them NaN, don't convert to binary.
                min_unique = np.min(unique)
                if type(min_unique) != str:
                    if np.isnan(min_unique):
                        self._add_numerical(features[col], aggregate_type=aggregate_type)
                        continue
                # Add as binary.
                self._add_binary(features[col], aggregate_type=aggregate_type)
                continue

            # Check for and add numerical features.
            if is_numeric_dtype(features[col]):
                self._add_numerical(features[col], aggregate_type=aggregate_type)
                continue

            # Check for (non-binary valued) string types.
            if is_string_dtype(features[col]) and len(unique) < 100:
                self._add_categorical(features[col], aggregate_type=aggregate_type)
                continue

            LOGGER.warning("Unsure about column %s data type, will not be added", col)

    def impute_features(
        self, static_imputer: Imputer, temporal_imputer: Imputer
    ) -> None:
        """Impute missing values in features.

        Parameters
        ----------
        static_imputer: Imputer
            Imputation options for static features.
        temporal_imputer: Imputer
            Imputation options for temporal features.

        """
        self.features[STATIC] = impute_features(
            self.features[STATIC], imputer=static_imputer
        )
        self.features[TEMPORAL] = impute_features(
            self.features[TEMPORAL], imputer=temporal_imputer
        )

    def plot_features(
        self,
        encounter_id: Optional[int] = None,
        aggregate_type: Optional[str] = TEMPORAL,
        names: Optional[Union[str, list]] = None,
    ) -> None:
        """Plot features.

        High-level plotting function for features. Can be used
        to plot static or temporal features based on arguments.

        Parameters
        ----------
        encounter_id: int, optional
            Encounter ID.
        aggregate_type: str
            'static' or 'temporal' features to plot.
        names: list or str, optional
            Names of features to plot.

        """
        if aggregate_type == TEMPORAL:
            if encounter_id:
                plot_temporal_features(self.features[TEMPORAL].loc[encounter_id], names)
        if aggregate_type == STATIC:
            plot_histogram(self.features[STATIC], names)

    def _drop_cols(self, cols: list) -> None:
        """Drop columns.

        Parameters
        ----------
        cols: list
            List of column names to drop.

        """
        assert cols is not None
        static_drops = [col for col in cols if col in self.static.columns]
        temporal_drops = [col for col in cols if col in self.temporal.columns]
        self.features[STATIC].drop(static_drops, axis=1, inplace=True)
        self.features[TEMPORAL].drop(temporal_drops, axis=1, inplace=True)

        for col in cols:
            del self.meta[col]

    def _drop_categorical(self, names: list) -> None:
        """Drop categorical groups of features.

        Parameters
        ----------
        names: list
            List of names of categorical columns to drop.

        """
        # Find which corresponding group columns to drop.
        drop_group_cols = [
            col
            for col in self.names
            if self.meta[col].feature_type == BINARY and self.meta[col].group in names
        ]

        # Drop corresponding columns.
        self._drop_cols(drop_group_cols)

    def drop_features(self, names: Union[str, list, set]) -> None:
        """Drop features.

        Parameters
        ----------
        names: str or list or set
            Name(s) of features to drop.

        """
        # Find feature columns to drop.
        names = set([names]) if isinstance(names, str) else set(names)
        drop_cols = names.intersection(set(self.names))
        remaining = names.difference(drop_cols)

        # Find categorical groups to drop.
        all_groups = [m.group for m in self.meta.values() if m.feature_type == BINARY]
        drop_groups = {c for c in remaining if c in all_groups}

        # Abort if not all the names are some column or categorical group.
        remaining = remaining.difference(drop_groups)
        if len(remaining) != 0:
            raise ValueError(f"Cannot drop non-existent features: {remaining}")

        # Drop columns.
        self._drop_cols(list(drop_cols))

        # Drop categorical group columns.
        self._drop_categorical(list(drop_groups))

    def save(self, folder_path: str, file_name: str) -> None:
        """Save features data in Parquet format.

        Parameters
        ----------
        folder_path: str
            Path to directory where the file can be saved.
        file_name: str
            Name of file (stored as 'static' and 'temporal').
            Extension will be .gzip.

        """
        if isinstance(self.features[STATIC], pd.DataFrame):
            os.makedirs(folder_path, exist_ok=True)
            save_file_name = file_name + "_" + STATIC
            save_path = os.path.join(folder_path, save_file_name + ".gzip")
            LOGGER.info("Saving static features to %s", save_path)
            self.features[STATIC].to_parquet(save_path)
        if isinstance(self.features[TEMPORAL], pd.DataFrame):
            os.makedirs(folder_path, exist_ok=True)
            save_file_name = file_name + "_" + TEMPORAL
            save_path = os.path.join(folder_path, save_file_name + ".gzip")
            LOGGER.info("Saving temporal features to %s", save_path)
            self.features[TEMPORAL].to_parquet(save_path)

    def load(self, folder_path: str, file_name: str) -> None:
        """Load features data from recognised compressed Parquet format.

        Parameters
        ----------
        folder_path: str
            Path to directory with files to load from.
        file_name: str
            Name of file ('static' and 'temporal' will both be attempted to load).
            Extension should be .gzip.

        """
        LOGGER.info("Loading features from file...")
        load_file_name = file_name + "_" + STATIC
        load_path = os.path.join(folder_path, load_file_name + ".gzip")
        if os.path.isfile(load_path):
            LOGGER.info("Found file to load for static features...")
            LOGGER.info("Successfully loaded static features from file...")
            self.add_features(pd.read_parquet(load_path))
        load_file_name = file_name + "_" + TEMPORAL
        load_path = os.path.join(folder_path, load_file_name + ".gzip")
        if os.path.isfile(load_path):
            LOGGER.info("Found file to load for temporal features...")
            self.add_features(pd.read_parquet(load_path))
            LOGGER.info("Successfully loaded temporal features from file...")
