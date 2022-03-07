"""Feature handling for automatic feature creation from processed data."""


import logging
import abc
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from codebase_ops import get_log_file_path

from cyclops.processors.constants import (
    FEATURE_TYPE,
    NUMERIC,
    BINARY,
    CATEGORICAL_BINARY,
    NORMALIZATION_METHOD,
    GROUP,
    STANDARD,
    MIN_MAX,
)
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

    """

    def __init__(self, feature_type: str) -> None:
        """Instantiate.

        Parameters
        ----------
        feature_type : str
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        """
        self.feature_type = feature_type

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

    def __init__(self, feature_type: str = BINARY, group: Optional[str] = None) -> None:
        """Instantiate.

        Parameters
        ----------
        feature_type : str
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        group: str, optional
            Name of group of feature, incase it belongs to a categorical group.
        """
        super().__init__(feature_type)

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
        normalization_method: Optional[str] = STANDARD,
    ):
        """Instantiate.

        Parameters
        ----------
        feature_type : str, optional
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        normalization_method: str, optional
            Name of normalization method, 'standard', 'min-max' or None.
        """
        super().__init__(feature_type)
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
    """Feature handler class."""

    def __init__(self, features: Optional[pd.DataFrame] = None) -> None:
        """Instantiate."""
        self.meta: list = []
        if features is None:
            self.features = pd.DataFrame()
        else:
            self.add_features(features)

    @property
    def unscaled(self) -> pd.DataFrame:
        """Return unscaled features."""
        return self.features

    @property
    def scaled(self) -> pd.DataFrame:
        """Scale and return scaled dataframe."""
        return self._scale()

    @property
    def names(self) -> list:
        """Access as attribute, feature names.

        Returns
        -------
        list
            List of feature names.
        """
        feature_names: list = []
        if self.features:
            feature_names = list(self.features.columns)
        return feature_names

    @property
    def types(self) -> list:
        """Access as attribute, feature types names.

        Note: These are built-in feature names.

        Returns
        -------
        list
            Feature type names.
        """
        return [f.feature_type for f in self.meta]

    def _scale(self) -> pd.DataFrame:
        """Apply scaling."""
        features = self.features.copy(deep=True)
        for i, col in enumerate(self.features.columns):
            if self.meta[i].feature_type == NUMERIC:
                features[col] = self.meta[i].scale(features[col])
        return features

    def extract_features(self, names: list) -> pd.DataFrame:
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
        # Convert to a list if extracting a single feature
        names = [names] if isinstance(names, str) else names

        # Ensure all features exist
        inter = set(names).intersection(set(self.features.columns))
        if len(inter) != len(names):
            raise ValueError(f"Features {inter} do not exist.")

        return self.features[names]

    def _add_feature(
        self,
        series: pd.Series,
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
        meta = feature_meta(**init_kwargs)
        series = meta.parse(series, **parse_kwargs)

        # Add to features.
        self.features = pd.concat([self.features, series], axis=1)

        # Add to meta.
        self.meta.append(meta)

    def _add_binary(self, series: pd.Series, group: Optional[str] = None) -> None:
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
        self._add_feature(series, BinaryFeatureMeta, init_kwargs, parse_kwargs)

    def _add_numeric(self, series: pd.Series, normalization_method=STANDARD) -> None:
        """Add numeric feature.

        Parameters
        ----------
        series: pandas.Series
            Feature column.
        """
        init_kwargs: Dict[str, Any] = {
            FEATURE_TYPE: NUMERIC,
            NORMALIZATION_METHOD: normalization_method,
        }
        parse_kwargs: Dict = {}
        self._add_feature(series, NumericFeatureMeta, init_kwargs, parse_kwargs)

    def _add_categorical(
        self,
        series: pd.Series,
    ) -> None:
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
            self._add_binary(features[col], group=series.name)

    def add_features(self, features: pd.DataFrame) -> None:
        """Add features.

        Parameters
        ----------
        features: pandas.DataFrame
            Features to add.

        """
        if isinstance(features, pd.DataFrame):
            self.features = pd.DataFrame(index=features.index)
        else:
            raise ValueError("input to feature handler must be a pandas.DataFrame.")

        # Attempt to turn any possible columns to numeric.
        features = _attempt_to_numeric(features)

        # Infer other column types.
        features = features.infer_objects()

        for col in features:
            unique = np.unique(features[col].values)

            # If all values are NaN, feature is dropped.
            if len(unique) == 1 and np.isnan(unique[0]):
                LOGGER.warning("Feature %s has all NaNs, will not be added.", col)
                continue

            # Check if it can be represented as binary.
            # (Numeric or string alike)
            if len(unique) == 2:
                # Add as binary.
                self._add_binary(features[col])
                continue

            # Check for and add numerical features.
            if is_numeric_dtype(features[col]):
                self._add_numeric(features[col])

            # Check for (non-binary valued) string types.
            elif is_string_dtype(features[col]):
                # Don't parse columns with too many unique values.
                if len(unique) > 100:
                    raise ValueError(f"Failed to parse feature {col}")
                self._add_categorical(features[col])
                continue

            else:
                raise ValueError("Unsure about column data type.")

    def _drop_cols(self, cols: list) -> None:
        """Drop columns.

        Parameters
        ----------
        cols: list
            List of columns to drop.
        """
        assert cols is not None
        # Drop columns.
        col_inds = [self.features.columns.get_loc(c) for c in cols]
        self.features.drop(self.features.columns[col_inds], axis=1, inplace=True)

        for i in sorted(col_inds, reverse=True):
            del self.meta[i]

    def _drop_categorical(self, names: list) -> None:
        """Drop categorical groups of features.

        Parameters
        ----------
        names: list
            List of names of categorical columns to drop.
        """
        # Find which corresponding group columns to drop.
        drop_group_cols = [
            c
            for i, c in enumerate(self.features.columns)
            if self.meta[i].feature_type == CATEGORICAL_BINARY
            and self.meta[i].group in names
        ]

        # Drop corresponding columns.
        self._drop_cols(drop_group_cols)

    def drop_features(self, names: Union[str, list, set]) -> None:
        """Drop features.

        Parameters
        ----------
        names: Union[str, list, set]
            Name(s) of features to drop.
        """
        # Find feature columns to drop.
        names = set([names]) if isinstance(names, str) else set(names)
        drop_cols = names.intersection(set(self.features.columns))
        remaining = names.difference(drop_cols)

        # Find categorical groups to drop.
        all_groups = [
            m.group for m in self.meta if m.feature_type == CATEGORICAL_BINARY
        ]
        drop_groups = {c for c in remaining if c in all_groups}

        # Abort if not all the names are some column or categorical group.
        remaining = remaining.difference(drop_groups)
        if len(remaining) != 0:
            raise ValueError(f"Cannot drop non-existent features: {remaining}")

        # Drop columns.
        self._drop_cols(list(drop_cols))

        # Drop categorical group columns.
        self._drop_categorical(list(drop_groups))
