"""Feature normalization."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cyclops.processors.constants import MIN_MAX, STANDARD
from cyclops.processors.util import has_columns, has_range_index
from cyclops.utils.common import to_list_optional


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
        method_map = {STANDARD: StandardScaler, MIN_MAX: MinMaxScaler}

        # Raise an exception if the method string is not recognized.
        if method not in method_map:
            options = ", ".join(["'" + k + "'" for k in method_map])
            raise ValueError(
                f"Method '{method}' is invalid, must be in: {', '.join(options)}."
            )

        self.scaler = method_map[method]()

    def fit(self, series: pd.Series) -> None:
        """Fit the scaler.

        Parameters
        ----------
        series: pandas.Series
            Series with values over which to fit.

        """
        self.scaler.fit(series.values.reshape(-1, 1))

    def transform(self, series: pd.Series) -> pd.Series:
        """Apply normalization.

        Parameters
        ----------
        series: pandas.Series
            Input series

        Returns
        -------
        pandas.Series
            Normalized series.

        """
        return pd.Series(
            np.squeeze(self.scaler.transform(series.values.reshape(-1, 1))),
            index=series.index,
        )

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Apply inverse normalization.

        Parameters
        ----------
        series: pandas.Series
            Input series

        Returns
        -------
        pandas.Series
            Inversely normalized series.

        """
        return pd.Series(
            np.squeeze(self.scaler.inverse_transform(series.values.reshape(-1, 1))),
            index=series.index,
        )

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

            return group

        # Over columns
        if self.by is None:
            for col in self.normalizer_map.keys():
                normalizer = self.normalizers.iloc[0][col]
                data[col] = getattr(normalizer, method)(data[col])
        # Over groups
        else:
            data = data.reset_index()
            data.set_index(self.by, inplace=True)
            grouped = data.groupby(self.by)
            data = grouped.apply(transform_group).reset_index()
            data.sort_values("index", inplace=True)
            data.drop("index", axis=1, inplace=True)

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
