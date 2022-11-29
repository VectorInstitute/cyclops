"""Feature normalization."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cyclops.process.constants import MIN_MAX, STANDARD
from cyclops.process.util import has_columns, has_range_index
from cyclops.utils.common import to_list_optional
from cyclops.utils.indexing import index_axis

METHOD_MAP = {STANDARD: StandardScaler, MIN_MAX: MinMaxScaler}


class SklearnNormalizer:
    """Sklearn normalizer wrapper.

    Attributes
    ----------
    method: str
        Name of normalization method.
    scaler: sklearn.preprocessing.StandardScaler or sklearn.preprocessing.MinMaxScaler
        Sklearn scaler object.

    """

    def __init__(self, method: str):
        """Initialize.

        Parameters
        ----------
        method: str
            String specifying the type of Sklearn scaler.

        """
        self.method = method

        # Raise an exception if the method string is not recognized.
        if method not in METHOD_MAP:
            options = ", ".join(["'" + k + "'" for k in METHOD_MAP])
            raise ValueError(
                f"Method '{method}' is invalid, must be in: {', '.join(options)}."
            )

        self.scaler = METHOD_MAP[method]()

    def __repr__(self):
        """Repr method.

        Returns
        -------
        str
            The normalization method name.

        """
        return self.method

    def fit(self, data: Union[np.ndarray, pd.Series]) -> None:
        """Fit the scaler.

        Parameters
        ----------
        data: numpy.ndarray or pandas.Series
            Data over which to fit.

        """
        # Ignore errors encountered when with null values
        # Save previous settings to later reset
        old_settings = np.seterr(divide="ignore", invalid="ignore")

        if isinstance(data, pd.Series):
            self.scaler.fit(data.values.reshape(-1, 1))

        elif isinstance(data, np.ndarray):
            if len(data.shape) != 1:
                raise ValueError("Data must be a 1D array.")

            self.scaler.fit(data.reshape(-1, 1))

        else:
            raise ValueError(
                "Data must be a pandas.Series or 1-dimensional numpy.ndarray"
            )

        # Reset to old settings
        np.seterr(**old_settings)

    def _transform_by_method(
        self,
        data: Union[np.ndarray, pd.Series],
        method: str,
    ) -> Union[np.ndarray, pd.Series]:
        """Apply a method on the scaler.

        If a numpy.ndarray is given, a numpy.ndarray is returned. Similarly, if a
        pandas.Series is given, a pandas.Series is returned.

        Parameters
        ----------
        data: numpy.ndarray or pandas.Series
            Input data.
        method: str
            Name of the method to apply.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Transformed data.

        """
        method_fn = getattr(self.scaler, method)
        # Ignore errors encountered when with null values
        # Save previous settings to later reset
        old_settings = np.seterr(divide="ignore", invalid="ignore")

        if isinstance(data, pd.Series):
            transformed = pd.Series(
                np.squeeze(method_fn(data.values.reshape(-1, 1))),
                index=data.index,
            )
        elif isinstance(data, np.ndarray):
            if len(data.shape) != 1:
                raise ValueError("Data must be a 1D array.")

            transformed = np.squeeze(method_fn(data.reshape(-1, 1)))

        else:
            raise ValueError(
                "Data must be a pandas.Series or 1-dimensional numpy.ndarray"
            )

        # Reset to old settings
        np.seterr(**old_settings)

        return transformed

    def transform(
        self, data: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        """Apply normalization.

        If a numpy.ndarray is given, a numpy.ndarray is returned. Similarly, if a
        pandas.Series is given, a pandas.Series is returned.

        Parameters
        ----------
        data: numpy.ndarray or pandas.Series
            Input data.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Normalized data.

        """
        return self._transform_by_method(data, "transform")

    def inverse_transform(
        self, data: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        """Apply inverse normalization.

        If a numpy.ndarray is given, a numpy.ndarray is returned. Similarly, if a
        pandas.Series is given, a pandas.Series is returned.

        Parameters
        ----------
        data: numpy.ndarray or pandas.Series
            Input data.

        Returns
        -------
        numpy.ndarray or pandas.Series
            Inversely normalized data.

        """
        return self._transform_by_method(data, "inverse_transform")


class GroupbyNormalizer:
    """Perform normalization over a DataFrame.

    Optionally normalize over specified groups rather than directly over columns.

    Attributes
    ----------
    normalizer_map: dict
        A map from the column name to the type of normalization, e.g.,
        {"event_values": "standard"}
    by: str or list of str, optional
        Columns to groupby, which affects how values are normalized.
    normalizers: dict
        The normalizer object information, where each column/group has a row
        with a normalization object.

    """

    def __init__(
        self,
        normalizer_map: dict,
        by: Optional[Union[str, List[str]]] = None,  # pylint: disable=invalid-name
    ) -> None:
        """Initialize."""
        features = set(normalizer_map.keys())

        # Check for duplicated occurences of features in the map.
        if len(list(normalizer_map.keys())) != len(features):
            raise ValueError("Cannot specify the same feature more than once.")

        self.normalizer_map = normalizer_map
        self.by = to_list_optional(by)  # pylint: disable=invalid-name
        self.normalizers = None

    def get_map(self) -> dict:
        """Get normalization mapping from features to type.

        Returns
        -------
        dict
            Normalization map.

        """
        return self.normalizer_map

    def get_by(self) -> Optional[List]:
        """Get groupby columns.

        Returns
        -------
        dict
            Groupby columns.

        """
        return self.by

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the normalizing objects.

        Parameters
        ----------
        data: pandas.DataFrame
            Data over which to fit.

        """
        if not has_range_index(data):
            raise ValueError(
                "DataFrame required to have a range index. Try resetting the index."
            )

        def get_normalizer_for_group(group: pd.DataFrame):
            cols = []
            data = []
            for col, method in self.normalizer_map.items():
                if isinstance(method, str):
                    normalizer = SklearnNormalizer(method)
                else:
                    raise ValueError(
                        """Must specify a string for the normalizer type.
                        Passing normalizer objects not yet supported."""
                    )

                normalizer.fit(group[col])
                cols.append(col)
                data.append(normalizer)

            return pd.DataFrame([data], columns=cols)

        if self.by is None:
            self.normalizers = get_normalizer_for_group(data)
        else:
            has_columns(data, self.by, raise_error=True)
            grouped = data.groupby(self.by)
            self.normalizers = grouped.apply(get_normalizer_for_group).droplevel(
                level=-1
            )

    def _transform_by_method(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply a method from the normalizer object to the data.

        Parameters
        ----------
        data: pandas.DataFrame
            Data to transform.
        method: str
            Name of normalizer method to apply.

        Returns
        -------
        pandas.DataFrame
            The data with the method applied.

        """
        if self.normalizers is None:
            raise ValueError("Must first fit the normalizers.")

        if not has_range_index(data):
            raise ValueError(
                "DataFrame required to have a range index. Try resetting the index."
            )

        def transform_group(group):
            for col in self.normalizer_map.keys():
                # Get normalizer object and transform
                normalizer = self.normalizers.loc[group.index.values[0]][col]
                group[col] = getattr(normalizer, method)(group[col])
            return group.reset_index()

        # Over columns
        if self.by is None:
            for col in self.normalizer_map.keys():
                normalizer = self.normalizers.iloc[0][col]
                data[col] = getattr(normalizer, method)(data[col])
        # Over groups
        else:
            # WARNING: The indexing here is very confusing, where if not done properly
            # the function will hang without explanation... devs be warned!
            assert data.index.name is None or data.index.name == "index"
            assert "index" not in data.columns
            data = data.reset_index()

            data.set_index(self.by, inplace=True)
            grouped = data.groupby(self.by, as_index=False, sort=False)
            data = grouped.apply(transform_group)
            data = data.reset_index(drop=True)
            data = data.set_index("index")
            data = data.sort_index()

        return data

    def transform(self, data: pd.DataFrame):
        """Normalize the data.

        Parameters
        ----------
        data: pandas.DataFrame
            Data to transform.

        Returns
        -------
        pandas.DataFrame
            The normalized data.

        """
        return self._transform_by_method(data, "transform")

    def inverse_transform(self, data: pd.DataFrame):
        """Inversely normalize the data.

        Parameters
        ----------
        data: pandas.DataFrame
            Data to transform.

        Returns
        -------
        pandas.DataFrame
            The inversely normalized data.

        """
        return self._transform_by_method(data, "inverse_transform")


class VectorizedNormalizer:
    """Perform normalization over a NumPy array.

    Attributes
    ----------
    axis: int
        Axis over which to normalize.
    normalizer_map: dict
        Mapping from an index to normalization type, e.g., {"eventA": "standard"}.
    normalizers: dict
        Mapping from an index to a normalizer object.
    is_fit: bool
        Whether or not the normalizers are fit.

    """

    def __init__(
        self,
        axis: int,
        normalizer_map: dict,
    ) -> None:
        """Initialize."""
        self.axis = axis
        self.normalizer_map: Dict[Any, str] = normalizer_map
        self.normalizers: Dict[Any, SklearnNormalizer] = {}
        self.is_fit: bool = False

    def get_map(self) -> Optional[dict]:
        """Get normalization mapping from features to type.

        Returns
        -------
        dict
            Normalization map.

        """
        return self.normalizer_map

    def _check_missing(self, index_map: Dict[str, int]) -> None:
        """Check if any of the normalizer_map features are missing a given index_map.

        Parameters
        ----------
        index_map: dict
            Map from feature name to index in the normalizer's given axis.

        """
        if self.normalizer_map is None:
            raise ValueError("normalizer_map is not set. Cannot check missing.")

        missing = set(self.normalizer_map.keys()) - set(index_map.keys())
        if len(missing) != 0:
            raise ValueError(f"Missing features {', '.join(missing)} in the data.")

    def fit(self, data: np.ndarray, index_map: Dict[str, int]) -> None:
        """Fit the normalizing objects.

        Parameters
        ----------
        data: numpy.ndarray
            Data over which to fit.
        index_map: dict
            Map from feature name to index in the normalizer's given axis.

        """
        self._check_missing(index_map)

        for feat, method in self.normalizer_map.items():
            if isinstance(method, str):
                normalizer = SklearnNormalizer(method)
            else:
                raise ValueError(
                    """Must specify a string for the normalizer type.
                    Passing normalizer objects not yet supported."""
                )

            ind = index_map[feat]
            values = data[index_axis(ind, self.axis, data.shape)]
            values = values.flatten()
            normalizer.fit(values)
            self.normalizers[feat] = normalizer

        self.is_fit = True

    def _transform_by_method(
        self, data: np.ndarray, index_map: Dict[str, int], method: str
    ) -> np.ndarray:
        """Apply a method from the normalizer object to the data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to transform.
        index_map: dict
            Map from feature name to index in the normalizer's given axis.
        method: str
            Name of normalizer method to apply.

        Returns
        -------
        numpy.ndarray
            The data with the method applied.

        """
        if not self.normalizers:
            raise ValueError("Must first fit the normalizers.")

        self._check_missing(index_map)

        for feat, normalizer in self.normalizers.items():
            ind = index_map[feat]
            data_indexing = index_axis(ind, self.axis, data.shape)
            values = data[data_indexing]
            prev_shape = values.shape
            values = values.flatten()
            normalized = getattr(normalizer, method)(values).reshape(prev_shape)
            data[data_indexing] = normalized

        return data

    def transform(self, data: np.ndarray, index_map: Dict[str, int]):
        """Normalize the data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to transform.
        index_map: dict
            Map from feature name to index in the normalizer's given axis.

        Returns
        -------
        numpy.ndarray
            The normalized data.

        """
        if not self.is_fit:
            raise ValueError("Normalizer has not been fit.")

        return self._transform_by_method(data, index_map, "transform")

    def inverse_transform(self, data: np.ndarray, index_map: Dict[str, int]):
        """Inversely normalize the data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to transform.
        index_map: dict
            Map from feature name to index in the normalizer's given axis.

        Returns
        -------
        numpy.ndarray
            The inversely normalized data.

        """
        if not self.is_fit:
            raise ValueError("Normalizer has not been fit.")

        return self._transform_by_method(data, index_map, "inverse_transform")

    def set_normalizers(self, normalizers: Dict[str, Any]) -> None:
        """Directly set the normalizer objects rather than fitting.

        Parameters
        ----------
        normalizers: dict
            Mapping from an index to a normalizer object.

        """
        if set(normalizers.keys()) - set(self.normalizer_map.keys()):
            raise ValueError("normalizers has keys not existing in the normalizer map.")
        self.normalizers = normalizers
        self.is_fit = True

    def subset(self, indexes: np.ndarray) -> VectorizedNormalizer:
        """Subset the normalizers and return this new VectorizedNormalizer.

        Parameters
        ----------
        indexes: numpy.ndarray
            Indexes to keep.

        """
        normalizer_map = copy.deepcopy(self.normalizer_map)

        keys = np.array(list(normalizer_map.keys()))
        keep = np.in1d(keys, indexes)

        keep_keys = keys[keep]
        normalizer_map = {
            key: val for key, val in normalizer_map.items() if key in keep_keys
        }

        normalizer = VectorizedNormalizer(self.axis, normalizer_map=normalizer_map)

        if self.is_fit:
            normalizers = copy.deepcopy(self.normalizers)
            normalizers = {
                key: val for key, val in normalizers.items() if key in keep_keys
            }
            normalizer.set_normalizers(normalizers)

        return normalizer
