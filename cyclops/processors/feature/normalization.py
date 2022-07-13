"""Feature normalization."""

from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cyclops.processors.constants import MIN_MAX, STANDARD
from cyclops.processors.util import has_columns, has_range_index
from cyclops.utils.common import to_list_optional

METHOD_MAP = {STANDARD: StandardScaler, MIN_MAX: MinMaxScaler}

def index_axis(ind: int, axis: int, shape: Tuple) -> Tuple:
    index = [slice(None)]*len(shape)
    index[axis] = ind
    return tuple(index)

def index_axis_ranged(start: int, stop: int, axis:int, shape: Tuple) -> Tuple:
    index = [slice(None)]*len(shape)
    index[axis] = slice(start, stop)
    return tuple(index)
    
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

    def fit(self, data: Union[np.ndarray, pd.Series]) -> None:
        """Fit the scaler.

        Parameters
        ----------
        data: numpy.ndarray or pandas.Series
            Data over which to fit.

        """
        # Series
        if isinstance(data, pd.Series):
            self.scaler.fit(data.values.reshape(-1, 1))
            return
        
        # Array
        if len(data.shape) != 1:
            raise ValueError("Data must be a 1D array.")
        self.scaler.fit(data.reshape(-1, 1))

    def transform(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
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
        # Series
        if isinstance(data, pd.Series):
            return pd.Series(
                np.squeeze(self.scaler.transform(data.values.reshape(-1, 1))),
                index=data.index,
            )
        
        # Array
        #if len(data.shape) != 1:
        #    raise ValueError("Data must be a 1D array.")
        
        return np.squeeze(self.scaler.transform(data.reshape(-1, 1)))
        

    def inverse_transform(self, data: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
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
        # Series
        if isinstance(data, pd.Series):
            return pd.Series(
                np.squeeze(self.scaler.inverse_transform(data.values.reshape(-1, 1))),
                index=data.index,
            )
        
        # Array
        #if len(data.shape) != 1:
        #    raise ValueError("Data must be a 1D array.")
        
        return np.squeeze(self.scaler.transform(data.reshape(-1, 1)))

    def __repr__(self):
        """Repr method.

        Returns
        -------
        str
            The normalization method name.

        """
        return self.method


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
    normalizers: pandas.DataFrame
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
            if data.index.name == None:
                data = data.reset_index(drop=True)
            else:
                data = data.reset_index()
            
            data.set_index(self.by, inplace=True)
            grouped = data.groupby(self.by, as_index=False, sort=False)
            data = grouped.apply(transform_group)
            data = data.reset_index(drop=True)
            #data = data.set_index("index")
            #data = data.sort_index()

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
    normalizer_map: dict
        A map from the column name to the type of normalization, e.g.,
        {"event_values": "standard"}
    normalizers: pandas.DataFrame
        The normalizer object information, where each column/group has a row
        with a normalization object.

    """

    def __init__(
        self,
        axis: int,
        normalization_method: Optional[str] = None,
        normalizer_map: Optional[dict] = None,
    ) -> None:
        """Initialize."""
        if normalization_method is None and normalizer_map is None:
            raise ValueError(
                ("Must specify normalization_method to normalize all features "
                 "with the same method, or normalization_map to map specific "
                 "indices to normalization methods.")
            )
        if normalization_method is not None and normalizer_map is not None:
            raise ValueError(
                "Cannot specify both normalization_method and normalization_map."
            )
        
        self.axis = axis
        self.normalization_method = normalization_method
        self.normalizer_map = normalizer_map
        self.normalizers = None

    def get_map(self) -> dict:
        """Get normalization mapping from features to type.

        Returns
        -------
        dict
            Normalization map.

        """
        return self.normalizer_map
    
    def _check_missing(self, feat_map: Dict[str, int]) -> None:
        """Check if any of the normalizer_map features are missing a given feat_map.
        
        Parameters
        ----------
        feat_map: dict
            Map from feature name to index in the normalizer's given axis.

        """
        missing = set(self.normalizer_map.keys()) - set(feat_map.keys())
        if len(missing) != 0:
            raise ValueError(f"Missing features {', '.join(missing)} in the data.")
    
    def fit(self, data: np.ndarray, feat_map: Dict[str, int]) -> None:
        """Fit the normalizing objects.

        Parameters
        ----------
        data: pandas.DataFrame
            Data over which to fit.
        feat_map: dict
            Map from feature name to index in the normalizer's given axis.

        """
        if self.normalizer_map is None:
            # Use the same normalization method for all features
            self.normalizer_map = {feat: self.normalization_method for feat in feat_map.keys()}
        else:
            self._check_missing(feat_map)
        
        self.normalizers = {}
        for feat, method in self.normalizer_map.items():
            if isinstance(method, str):
                normalizer = SklearnNormalizer(method)
            else:
                raise ValueError(
                    """Must specify a string for the normalizer type.
                    Passing normalizer objects not yet supported."""
                )
            
            ind = feat_map[feat]
            values = data[index_axis(ind, self.axis, data.shape)]
            print("values", values)
            values = values.flatten()
            print("values.shape", values.shape)
            normalizer.fit(values)
            self.normalizers[feat] = normalizer

    def _transform_by_method(self, data: np.ndarray, feat_map: Dict[str, int],  method: str) -> np.ndarray:
        """Apply a method from the normalizer object to the data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to transform.
        feat_map: dict
            Map from feature name to index in the normalizer's given axis.
        method: str
            Name of normalizer method to apply.

        Returns
        -------
        numpy.ndarray
            The data with the method applied.

        """
        if self.normalizers is None:
            raise ValueError("Must first fit the normalizers.")
            
        self._check_missing(feat_map)

        for feat, normalizer in self.normalizers.items():
            ind = feat_map[feat]
            data_indexing = index_axis(ind, self.axis, data.shape)
            values = data[data_indexing]
            prev_shape = values.shape
            print("feat\n", feat)
            print("values\n", values)
            values = values.flatten()
            normalized = getattr(normalizer, method)(values).reshape(prev_shape)
            print("normalized\n", normalized)
            data[data_indexing] = normalized
            print("\n")
        return data

    def transform(self, data: np.ndarray, feat_map: Dict[str, int]):
        """Normalize the data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to transform.
        feat_map: dict
            Map from feature name to index in the normalizer's given axis.

        Returns
        -------
        numpy.ndarray
            The normalized data.

        """
        return self._transform_by_method(data, feat_map, "transform")

    def inverse_transform(self, data: np.ndarray, feat_map: Dict[str, int]):
        """Inversely normalize the data.

        Parameters
        ----------
        data: numpy.ndarray
            Data to transform.
        feat_map: dict
            Map from feature name to index in the normalizer's given axis.

        Returns
        -------
        numpy.ndarray
            The inversely normalized data.

        """
        return self._transform_by_method(data, feat_map, "inverse_transform")
        