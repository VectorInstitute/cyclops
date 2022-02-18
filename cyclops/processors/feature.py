"""Feature containers for automatic feature creation from raw data."""


from abc import ABC

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def category_to_numeric(series, inplace=False, unique=None):
    """
    Takes a series and replaces its the values with the index
    of their value in the array's sorted, unique values.

    Parameters:
        series (pandas.Series): A Pandas series.
        inplace (bool): Whether to replace values in-place.
        unique (numpy.ndarray): The series' unique values may
            be optionally given if already calculated.
    """
    # Calculate unique values if not given
    if unique is None:
        unique = np.unique(series.values)
    unique.sort()

    # Create mapping from sorted unique values to index
    map_dict = dict()
    for i, u in enumerate(unique):
        map_dict[u] = i

    return series.replace(map_dict, inplace=inplace)


def is_string_type(arr):
    """
    Checks whether a NumPy array has a datatype which could
    possibly hold string values.

    Parameters:
        arr (numpy.ndarray): A NumPy array.

    Returns:
        (bool) Whether this array's datatype could
            possibly hold string values.
    """
    s = "".join([char for char in str(arr.dtype) if char.isalpha()])
    STR_TYPE = ["U", "O", "S", "str", "string", "object"]
    return s in STR_TYPE


class FeatureMeta(ABC):
    """
    An abstract feature class to act as parent for concrete feature classes.

    Attributes
    ----------
    name : str
        Feature name.
    feature_type : str
        The type of the feature, e.g., 'binary', 'numeric', or
            'categorical-binary'.

    Methods
    -------
    _scale(values):
        An identity function used to handle classes without scaling.

    _inverse_scale(values):
        An identity function used to handle classes without scaling.
    """

    def __init__(self, feature_type):
        self.feature_type = feature_type

    def parse(self, series):
        return series

    def _scale(self, values):
        """
        An identity function used to handle scaling for classes
        without scaling support.

        Parameters:
            values (numpy.ndarray): A 1-dimensional NumPy array.

        Returns:
            (numpy.ndarray): values array, unchanged.
        """
        return values

    def _inverse_scale(self, values):
        """
        An identity function used to handle inverse scaling for
        classes without scaling support.

        Parameters:
            values (numpy.ndarray): A 1-dimensional NumPy array.

        Returns:
            (numpy.ndarray): values array, unchanged.
        """
        return values


class BinaryFeatureMeta(FeatureMeta):
    """
    A class for handling binary features, i.e., with values 0, 1. Any
    acceptable inputs which can be converted to binary values, such
    as an array with only unique values 'A', 'B' will be converted to
    0, 1.
    """

    def __init__(self, feature_type="binary", group=None):
        super().__init__(feature_type)

        # Group is used to track the group of a binary categorical
        # variable
        self.group = group

    def parse(self, series):
        unique = np.unique(series.values)
        np.sort(unique)

        if len(unique) != 2:
            raise ValueError(
                "Binary features must have two unique values, e.g., [0, 1], ['A', 'B']."
            )

        # Convert strings to numerical binary values
        if not np.array_equal(unique, np.array([0, 1])):
            series = category_to_numeric(series, unique=unique)

        return series.astype(np.uint8)


class NumericFeatureMeta(FeatureMeta):
    """
    A class for handling numeric features, with normalization
    functionality.

    Methods
    -------
    _get_scaler_type(values):
        Returns a scaling object mapped from a string value.

    _scale(values):
        Scales a 1D array based on selected scaling object. If the
        scaler is none, it acts as an identity function.

    _scale(values):
        Inverses scaling a 1D array based on selected scaling object.
        If the scaler is none, it acts as an identity function.
    """

    def __init__(self, feature_type="numeric", normalization="standardize"):
        super().__init__(feature_type)
        self.normalization = normalization

    def parse(self, series, scale=True):
        # Create scaling object and scale data
        if self.normalization is None:
            self.scaler = None
        else:
            Scaler = self._get_scaler_type(self.normalization)
            self.scaler = Scaler().fit(series.values.reshape(-1, 1))

        if scale:
            return self.scale(series)

        return series

    def _get_scaler_type(self, normalization):
        """
        Returns a scaling object mapped from a string value.

        Parameters:
            normalization (str): A string specifying which scaler to return.

        Returns:
            (object): An sklearn.preprocessing scaling object.
        """
        scaler_map = {"standardize": StandardScaler, "minMax": MinMaxScaler}

        # Raise an exception if the normalization string is not recognized
        if not normalization in list(scaler_map.keys()):
            raise ValueError(
                "'{}' is invalid. Normalization input must be in None, {}".format(
                    normalization,
                    ", ".join(["'" + k + "'" for k in list(scaler_map.keys())]),
                )
            )

        # Otherwise, return the corresponding scaling object
        return scaler_map[normalization]

    def scale(self, series):
        """
        Scales a 1D array based on selected scaling object. If the
        scaler is none, it acts as an identity function.

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
        """
        Inverses scaling a 1D array based on selected scaling object.
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


class FeatureStore:
    def __init__(self, df=None, maintain_unscaled=False):
        # self.df = self.df_to_features(df) if df is not None else None
        self.df = None
        self.df_unscaled = None
        self.metas = []
        self.maintain_unscaled = maintain_unscaled

    @property
    def names(self):
        """
        Accessed an as attribute, this function returns feature names.

        Returns:
            (list<str>): Feature names.
        """
        return list(self.df.columns)

    @property
    def types(self):
        """
        Accessed an as attribute, this function returns feature type
        names. Note: These are built-in feature names, not NumPy's
        dtype feature names, for example.

        Returns:
            (list<str>): Feature type names.
        """
        return [f.feature_type for f in self.metas]

    def extract_features(self, names):
        """
        Extract features by name.

        Parameters:
            names (list<str>, str): Feature name(s).

        Returns:
            (pandas.DataFrame): Requested features formatted as a
                DataFrame.
        """
        # Convert to a list if extracting a single feature
        names = [names] if isinstance(names, str) else names

        # Ensure all features exist
        inter = set(names).intersection(set(self.df.columns))
        if len(inter) != len(names):
            raise ValueException(
                "Features {} do not exist.".format(
                    set(names).difference(set(self.df.columns))
                )
            )

        return df[names]

    def _values_expand(self, values):
        """
        Expands a 1-dimensional NumPy array to be 2D. This is necessary when
        handling a feature array which must be converted to a feature matrix
        with one feature.

        Parameters:
            values (numpy.ndarray): Feature vector or matrix.

        Returns:
            values viewed as 2D matrix if values was 1D feature vector,
            otherwise returns values without change.
        """
        if values.ndim == 1:
            return np.expand_dims(values, -1)
        return values

    def _assert_feature_length(self, df):
        """
        Asserts that the number of features in a dataset is the same
        as the number of features added here.

        Parameters:
            values (numpy.ndarray): Feature vector or matrix.
        """
        # If no features have been added, perform no check
        if df is None:
            return

        if len(df.columns) != len(self.df.columns):
            raise ValueError(
                "Number of features {} must match {}.".format(
                    len(df.columns), len(self.df.columns)
                )
            )

    def _assert_nrows(self, obj):
        """
        Asserts number of samples (rows) in a Pandas Series/
        DataFrame is the same as any previously added features.

        Parameters:
            obj (pandas.Series, pandas.DataFrame): Features.
        """
        if self.df is None:
            return

        if len(self.df) != len(values):
            raise ValueError("Number of rows must be the same for all features added.")

    def _handle_features_names(self, values, names):
        """
        Determines feature names either based on those inputted,
        or by generating names if none were given.

        Parameters:
            values (numpy.ndarray): An array of values, e.g., from
                df.values.
            names (list<str>, str): A list of names, or a string is
                accepted if adding a single feature.

        Returns:
            (list<str>): Parsed/generated feature names
        """

        # If names is not defined, come up with some default names
        df_len = 0 if self.df is None else len(self.df.columns)
        if names is None:
            names = np.arange(stop=values.shape[1]) + df_len
            return np.core.defchararray.add("feature", np.char.mod("%d", names))

        # Otherwise, check if a string, and if so,
        # assert there is just one feature being added
        if isinstance(names, str):
            assert values.shape[1] == 1
            return [names]

        if len(names) != values.shape[1]:
            raise ValueError("Length of names does not match length of features.")

        return names

    def _format_add(self, values, names):
        if isinstance(values, pd.DataFrame):
            return values

        df = pd.DataFrame(values)
        names = self._handle_features_names(df.values, names)
        df.columns = names
        return df

    def _add(self, df, FMeta, init_kwargs, parse_kwargs, attr_kwargs, unscaled=False):
        """
        An internal use function for prepping to add features
        of the same type.

        Parameters:
            df (pandas.DataFrame): Feature DataFrame to concat.
            FMeta (object): Feature metadata class.

        Returns:
            (numpy.ndarray): Prepared feature matrix.
            (list<str>): Prepared feature names.
        """
        if self.maintain_unscaled:
            df_org = df.copy(deep=True)

        # Add to features metadata
        metas = []
        for col in df:
            # If forcing unscaled
            if unscaled:
                if "scale" in parse_kwargs:
                    parse_kwargs["scale"] = False

            metas.append(FMeta(**init_kwargs))

            df[col] = metas[-1].parse(df[col], **parse_kwargs)

        # Add to features
        if not unscaled:
            self.df = pd.concat([self.df, df], axis=1)
            self.metas = np.concatenate([self.metas, metas])
        else:
            self.df_unscaled = pd.concat([self.df_unscaled, df], axis=1)

        # Maintain unscaled
        if self.maintain_unscaled and not unscaled:
            self._add(
                df_org, FMeta, init_kwargs, parse_kwargs, attr_kwargs, unscaled=True
            )

    def add_binary(self, values, names=None, feature_type="binary", group=None):
        """
        Adds binary features.

        Parameters:
            values (list, numpy.ndarray, pandas.Series,
                pandas.DataFrame): Feature object.
            names (list<str>, str): Feature name(s).
            feature_type (str): Feature type name.
        """
        # Input checking and preparation
        df = self._format_add(values, names)

        init_kwargs = {"feature_type": feature_type, "group": group}
        parse_kwargs = {}
        attr_kwargs = {}
        self._add(df, BinaryFeatureMeta, init_kwargs, parse_kwargs, attr_kwargs)

    def add_numeric(
        self,
        values,
        names=None,
        feature_type="numeric",
        normalization="standardize",
        scale=True,
    ):
        """
        Adds numeric features.

        Parameters:
            values (numpy.ndarray): Feature vector or matrix.
            names (list<str>, str): Feature name(s).
            feature_type (str): Feature type name.
        """
        # Input checking and preparation
        df = self._format_add(values, names)

        init_kwargs = {"feature_type": feature_type, "normalization": normalization}
        init_kwargs = {"feature_type": feature_type}
        parse_kwargs = {"scale": scale}
        attr_kwargs = {}
        self._add(df, NumericFeatureMeta, init_kwargs, parse_kwargs, attr_kwargs)

    def add_categorical(self, values, names=None, feature_type="categorical-binary"):
        """
        Adds categorical features.

        Parameters:
            values (numpy.ndarray): Feature vector or matrix.
            names (list<str>, str): Feature name(s).
            feature_type (str): Feature type name.
        """
        # Input checking and preparation
        df = self._format_add(values, names)

        for col in df.columns:
            vals = df[col].values

            unique = np.unique(vals)
            unique.sort()

            binary_names = np.core.defchararray.add(col + "-", unique.astype(str))

            # Categorical as indices
            a = category_to_numeric(df[col], unique=unique)

            # One hot encoding
            onehot = np.zeros((a.size, a.max() + 1))
            onehot[np.arange(a.size), a] = 1

            self.add_binary(
                onehot, names=binary_names, feature_type=feature_type, group=col
            )

    def _attempt_to_numeric(self, df):
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except:
                pass
        return df

    def add_features(self, values, names=None):
        if isinstance(values, pd.DataFrame):
            df = values
        elif isinstance(values, np.ndarray):
            names = self._handle_features_names(values, names=names)
            df = pd.DataFrame(values, columns=names)
        else:
            raise ValueError("values must be a pandas.DataFrame or numpy.ndarray.")

        # Attempt to turn any possible columns to numeric
        df = self._attempt_to_numeric(df)

        # Infer other column types
        # print(df.info(verbose=True))
        df = df.infer_objects()
        # print(df.info(verbose=True))

        for col in df:
            unique = np.unique(df[col].values)
            np.sort(unique)

            # Check if it can be represented as binary
            # (Numeric or string alike)
            if len(unique) == 2:
                # Add as binary
                self.add_binary(df[col].values, names=col)
                continue

            # Check for and add numerical features
            if is_numeric_dtype(df[col]):
                self.add_numeric(df[col].values, names=col)

            # Check for (non-binary valued) string types
            elif is_string_dtype(df[col]):
                # Don't parse columns with too many unique values
                if len(unique) > 100:
                    raise ValueError("Failed to parse feature {}".format(col))

                # Add as categorical
                self.add_categorical(df[col].values, names=col)
                continue

            else:
                raise ValueException("Unsure about column data type.")

    def remove_features(self, names):
        raise NotImplementedError()

        names = [names] if isinstance(names, str) else names
        drop_cols = set(names).intersection(set(self.df.columns))

        # Drop columns directly
        self.df.drop(list(drop_cols), axis=1, inplace=True)
        self.metas = None  # UPDATE METAS
        if self.maintain_unscaled:
            self.df_unscaled.drop(list(drop_cols), axis=1, inplace=True)

        # Check for additional, non-column drops, e.g.,
        # dropping a categorical group
        additional = set(names).difference(drop_cols)

        categorical = [
            m.group for m in self.metas if m.feature_type == "categorical-binary"
        ]

        drop_categorical = additional.intersection(categorical)

        # Drop columns directly
        for d in drop_categorical:
            cat = [c for c in categorical if c.group == d]

            self.df.drop(cat, axis=1, inplace=True)
            if self.maintain_unscaled:
                self.df_unscaled.drop(cat, axis=1, inplace=True)

        remaining = additional.difference(drop_categorical)
        if len(remaining) != 0:
            raise ValueError("Cannot drop non-existent features: {}".format(remaining))

    def scale(self, values):
        raise NotImplementedError()

    def inverse_scale(self, values):
        raise NotImplementedError()
