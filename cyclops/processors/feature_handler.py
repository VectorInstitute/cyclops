"""Feature handling for automatic feature creation from processed data."""


import abc
from abc import ABC
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler


NUMERIC = "numeric"
BINARY = "binary"
CATEGORICAL_BINARY = "categorical-binary"


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
    scaler_map = {"standard": StandardScaler, "min-max": MinMaxScaler}

    # Raise an exception if the normalization string is not recognized.
    if normalization_method not in list(scaler_map.keys()):
        raise ValueError(
            "'{}' is invalid. Normalization input must be in None, {}".format(
                normalization_method,
                ", ".join(["'" + k + "'" for k in list(scaler_map.keys())]),
            )
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

    map_dict = dict()
    for i, u in enumerate(unique):
        map_dict[u] = i

    if inplace:
        series.replace(map_dict, inplace=inplace)
        return series
    else:
        return series.replace(map_dict, inplace=inplace)


class FeatureMeta(ABC):
    """Abstract feature class to act as parent for concrete feature classes.

    Attributes
    ----------
    feature_type : str
        The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.

    """

    def __init__(self, feature_type) -> None:
        """Instantiate.

        Parameters
        ----------
        feature_type : str
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        """
        self.feature_type = feature_type

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

    def scale(self, series: pd.Series) -> pd.Series:
        """Scale feature column, returns input if not implemented in child class.

        Parameters
        ----------
        series: pandas.Series
            A 1-dimensional input feature column.

        Returns
        -------
        pandas.Series
            Scaled feature column, same if scaling is not implemented/used.
        """
        return series

    def inverse_scale(self, values):
        """Inverse scale feature column, returns input if not implemented in child class.

        Parameters
        ----------
        series: pandas.Series
            A 1-dimensional input feature column.

        Returns
        -------
        pandas.Series
            Inverse scaled feature column, same if scaling is not implemented/used.
        """
        return values


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
        # Convert strings to numerical binary values
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

    def __init__(self, feature_type=NUMERIC, normalization_method="standard"):
        """Instantiate.

        Parameters
        ----------
        feature_type : str
            The type of the feature, e.g., 'binary', 'numeric', or 'categorical-binary'.
        normalization_method: str, optional
            Name of normalization method, 'standard', 'min-max' or None.
        """
        super().__init__(feature_type)
        self.normalization_method = normalization_method

    def parse(self, series):
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
        if self.normalization_method is None:
            self.scaler = None
        else:
            Scaler = _get_scaler_type(self.normalization_method)
            self.scaler = Scaler().fit(series.values.reshape(-1, 1))

        return series

    def scale(self, series):
        """Scale a 1D array based on selected scaling object.

        If the scaler is none, it acts as an identity function.

        Parameters:
            values (numpy.ndarray): A 1-dimensional NumPy array.

        Returns:
            (numpy.ndarray): values array, scaled.
        """
        if self.scaler is None:
            return series

        return pd.Series(
            np.squeeze(self.scaler.transform(series.values.reshape(-1, 1)))
        )

    def inverse_scale(self, series):
        """Apply Inverse scaling to a 1D array based on selected scaling object.

        If the scaler is none, it acts as an identity function.

        Parameters:
            values (numpy.ndarray): A 1-dimensional NumPy array.

        Returns:
            (numpy.ndarray): values array, inverse scaled.
        """
        if self.scaler is None:
            return series

        return pd.Series(
            np.squeeze(self.scaler.inverse_transform(series.values.reshape(-1, 1)))
        )


class FeatureHandler:
    """Feature handler class."""

    def __init__(self, df=None):
        """Instantiate."""
        if df is None:
            self.df = None
        else:
            self.df = self.add_features(df)
        self.meta = []

    @property
    def df_unscaled(self):
        """Return unscaled dataframe."""
        return self.df

    @property
    def df_scaled(self):
        """Scale and return scaled dataframe."""
        return self._scale()

    @property
    def names(self):
        """Access as attribute, feature names.

        Returns
        -------
        (list<str>)
            Feature names.
        """
        return list(self.df.columns)

    @property
    def types(self):
        """Access as attribute, feature types names.

        Note: These are built-in feature names, not NumPy's
        dtype feature names, for example.

        Returns
        -------
        (list<str>)
            Feature type names.
        """
        return [f.feature_type for f in self.meta]

    def extract_features(self, names):
        """Extract features by name.

        Parameters
        ----------
        names: (list<str>, str)
            Feature name(s).

        Returns
        -------
        (pandas.DataFrame)
            Requested features formatted as a DataFrame.
        """
        # Convert to a list if extracting a single feature
        names = [names] if isinstance(names, str) else names

        # Ensure all features exist
        inter = set(names).intersection(set(self.df.columns))
        if len(inter) != len(names):
            raise ValueError(
                "Features {} do not exist.".format(
                    set(names).difference(set(self.df.columns))
                )
            )

        return self.df[names]

    def _add_feature(
        self,
        values: pd.Series,
        FMeta: abc.ABCMeta,
        init_kwargs: dict,
        parse_kwargs: dict,
    ) -> None:
        """Add feature (internal method).

        Parameters
        ----------
        values: pandas.Series
            Feature column to concat.
        FMeta: abc.ABCMeta
            Feature metadata class.
        init_kwargs: dict
            Init keyword arguments for feature meta.
        parse_kwargs: dict
            Parse keyword arguments for parsing feature using meta information.

        """
        meta = FMeta(**init_kwargs)
        values = meta.parse(values, **parse_kwargs)

        # Add to features.
        self.df = pd.concat([self.df, values], axis=1)

        # Add to meta.
        self.meta.append(meta)

    def _add_binary(self, values, group=None):
        """Add binary features.

        Parameters
        ----------
        values: pandas.Series
            Feature column.
        group: Optional[str]
            Feature type name.
        """
        init_kwargs = {"feature_type": BINARY, "group": group}
        parse_kwargs = {}
        self._add_feature(values, BinaryFeatureMeta, init_kwargs, parse_kwargs)

    def _add_numeric(self, values: pd.Series, normalization_method="standard"):
        """Add numeric feature.

        Parameters
        ----------
        values: pandas.Series
            Feature column.
        """
        init_kwargs: Dict[str, Any] = {
            "feature_type": NUMERIC,
            "normalization_method": normalization_method,
        }
        parse_kwargs: Dict = {}
        self._add_feature(values, NumericFeatureMeta, init_kwargs, parse_kwargs)

    def _add_categorical(
        self,
        values: pd.Series,
    ):
        """Add categorical feature.

        Parameters
        ----------
        values: pandas.Series
            Feature column.
        """
        unique = np.unique(values)

        binary_names = [
            values.name + "-" + unique_val for unique_val in unique.astype(str)
        ]
        numeric = _category_to_numeric(values, unique=unique)
        onehot = np.zeros((numeric.size, numeric.max() + 1))
        onehot[np.arange(numeric.size), numeric] = 1
        df = pd.DataFrame(onehot, index=values.index, columns=binary_names)

        for col in df:
            self._add_binary(df[col], group=values.name)

    def _attempt_to_numeric(self, df):
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
        return df

    def add_features(self, values, names=None):
        """Add features."""
        if isinstance(values, pd.DataFrame):
            df = values
            self.df = pd.DataFrame(index=df.index)
        else:
            raise ValueError("input to feature handler must be a pandas.DataFrame.")

        # Attempt to turn any possible columns to numeric.
        df = self._attempt_to_numeric(df)

        # Infer other column types.
        df = df.infer_objects()

        for col in df:
            unique = np.unique(df[col].values)

            # Check if it can be represented as binary.
            # (Numeric or string alike)
            if len(unique) == 2:
                # Add as binary.
                self._add_binary(df[col])
                continue

            # Check for and add numerical features.
            if is_numeric_dtype(df[col]):
                self._add_numeric(df[col])
            # Check for (non-binary valued) string types.
            elif is_string_dtype(df[col]):
                # Don't parse columns with too many unique values.
                if len(unique) > 100:
                    raise ValueError("Failed to parse feature {}".format(col))
                # Add as categorical.
                self._add_categorical(df[col])
                continue
            else:
                raise ValueError("Unsure about column data type.")

    def _drop_cols(self, cols):
        assert cols is not None
        # Drop columns
        col_inds = [self.df.columns.get_loc(c) for c in cols]
        self.df.drop(self.df.columns[col_inds], axis=1, inplace=True)

        for i in sorted(col_inds, reverse=True):
            del self.meta[i]

    def _drop_categorical(self, names):
        # Find which corresponding group columns to drop
        drop_group_cols = [
            c
            for i, c in enumerate(self.df.columns)
            if self.meta[i].feature_type == "categorical-binary"
            and self.meta[i].group in names
        ]

        # Drop corresponding columns
        self._drop_cols(drop_group_cols)

    def drop_features(self, names):
        """Drop features."""
        # Find feature columns to drop
        names = set([names]) if isinstance(names, str) else set(names)
        drop_cols = names.intersection(set(self.df.columns))
        remaining = names.difference(drop_cols)

        # Find categorical groups to drop
        all_groups = [
            m.group for m in self.meta if m.feature_type == "categorical-binary"
        ]
        drop_groups = set([c for c in remaining if c in all_groups])

        # Abort if not all the names are some column or
        # categorical group
        remaining = remaining.difference(drop_groups)
        if len(remaining) != 0:
            raise ValueError("Cannot drop non-existent features: {}".format(remaining))

        # Drop columns
        self._drop_cols(drop_cols)

        # Drop categorical group columns
        self._drop_categorical(list(drop_groups))

    def _scale(self):
        """Apply scaling."""
        df = self.df.copy(deep=True)
        for i, col in enumerate(self.df.columns):
            df[col] = self.meta[i].scale(df[col])
        return df
