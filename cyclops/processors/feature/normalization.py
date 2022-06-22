"""Feature normalization."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cyclops.processors.constants import MIN_MAX, STANDARD


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
            raise ValueError(f"'{method}' is invalid, must be in None, {options}")

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
        """Apply normalization, or scaling.

        Parameters
        ----------
        series: pandas.Series
            Input series

        Returns
        -------
        pandas.Series
            Scaled series.

        """
        return pd.Series(
            np.squeeze(self.scaler.transform(series.values.reshape(-1, 1))),
            index=series.index,
        )

    def inverse_transform(self, series: pd.Series) -> pd.Series:
        """Apply inverse normalization, or scaling.

        Parameters
        ----------
        series: pandas.Series
            Input series

        Returns
        -------
        pandas.Series
            Scaled series.

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
    """Perform normalization over a DataFrame, possibly over specific groups.

    Attributes
    ----------
    normalizer_map: dict
        A map from the column name to the type of normalization, e.g.,
        "event_values": "standard"
    normalizers: pandas.DataFrame
        Storing the normalizer object information, where each group has a row
        of scaling objects.
    by: str or list of str, optional
        Columns to groupby, affecting how values are normalized.

    """

    def __init__(
        self,
        normalizer_map: dict,
        by: Optional[Union[str, List[str]]] = None,  # pylint: disable=invalid-name
    ) -> None:
        """Initialize."""
        self.normalizer_map = normalizer_map
        self.by = by  # pylint: disable=invalid-name
        self.normalizers = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the normalizing objects.

        Parameters
        ----------
        data: pandas.DataFrame
            Data over which to fit.

        """
        # has_columns(data, self.by, raise_error=True)

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

        # has_columns(data, self.by, raise_error=True)

        def transform_group(group):
            for col in self.normalizer_map.keys():
                # Get normalizer object and transform
                normalizer = self.normalizers.loc[group.index.values[0]][col]
                group[col] = getattr(normalizer, method)(group[col])

            return group

        if self.by is None:
            for col in self.normalizer_map.keys():
                normalizer = self.normalizers.iloc[0][col]
                data[col] = getattr(normalizer, method)(data[col])
        else:
            data = data.reset_index()
            data.set_index(self.by, inplace=True)
            grouped = data.groupby(self.by)
            data = grouped.apply(transform_group).reset_index()

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
        """Inverse normalize the data.

        Parameters
        ----------
        data: pandas.DataFrame
            Data to transform.

        Returns
        -------
        pandas.DataFrame
            The inverse normalized data.

        """
        return self._transform_by_method(data, "inverse_transform")
