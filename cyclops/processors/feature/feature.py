"""Feature processing."""

import logging
from typing import Dict, List, Optional, Union

import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.plotter import plot_histogram, plot_temporal_features
from cyclops.processors.aggregate import Aggregator
from cyclops.processors.constants import (
    CATEGORICAL_INDICATOR,
    FEATURE_INDICATOR_ATTR,
    FEATURE_MAPPING_ATTR,
    FEATURE_META_ATTR_DEFAULTS,
    FEATURE_META_ATTRS,
    FEATURE_TARGET_ATTR,
    FEATURE_TYPE_ATTR,
    FEATURE_TYPES,
    NUMERIC,
    ORDINAL,
)
from cyclops.processors.feature.normalization import GroupbyNormalizer
from cyclops.processors.feature.type_handling import (
    infer_types,
    normalize_data,
    to_types,
)
from cyclops.processors.util import has_columns, to_range_index
from cyclops.utils.common import to_list, to_list_optional
from cyclops.utils.file import save_dataframe
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=get_log_file_path(), print_level="INFO", logger=LOGGER)


class FeatureMeta:
    """Feature metadata class.

    Attributes
    ----------
    feature_type: str
        Feature type.
    target: bool
        Whether the feature a target variable.
    indicator_of: str, optional
        If not None, the feature from which this indicator was generated.

    """

    def __init__(self, **kwargs) -> None:
        """Init."""
        # Feature type checking
        if FEATURE_TYPE_ATTR not in kwargs:
            raise ValueError("Must specify feature type.")

        if kwargs[FEATURE_TYPE_ATTR] not in FEATURE_TYPES:
            raise ValueError(
                f"""Feature type '{kwargs[FEATURE_TYPE_ATTR]}'
                not in {', '.join(FEATURE_TYPES)}."""
            )

        # Set attributes
        for attr in FEATURE_META_ATTRS:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, FEATURE_META_ATTR_DEFAULTS[attr])

        # Check for invalid parameters
        invalid_params = [kwarg for kwarg in kwargs if kwarg not in FEATURE_META_ATTRS]
        if len(invalid_params) > 0:
            raise ValueError(
                f"Invalid feature meta parameters {', '.join(invalid_params)}."
            )

    def get_type(self) -> str:
        """Get the feature type.

        Returns
        -------
        str
            Feature type.

        """
        return getattr(self, FEATURE_TYPE_ATTR)

    def is_target(self) -> bool:
        """Get whether the feature is a target.

        Returns
        -------
        bool
            Whether the feature is a target.

        """
        return getattr(self, FEATURE_TARGET_ATTR)

    def indicator_of(self) -> Optional[str]:
        """Get the name of an indicator's original categorical feature.

        Returns
        -------
        str, optional
            The categorical column from which an indicator was generated, or None if
            not a categorical indicator.

        """
        return getattr(self, FEATURE_INDICATOR_ATTR)

    def get_mapping(self) -> Optional[dict]:
        """Get the category value map for binary and ordinal categories.

        Returns
        -------
        dict, optional
            A mapping from the integer categories to the original values, or None if
            there is no mapping.

        """
        return getattr(self, FEATURE_MAPPING_ATTR)

    def update(self, meta: List[tuple]) -> None:
        """Update meta attributes.

        Parameters
        ----------
        meta: list of tuple
            List of tuples in the format (attribute name, attribute value).

        """
        for info in meta:
            setattr(self, *info)


class Features:
    """Features.

    Attributes
    ----------
    data: pandas.DataFrame
        Features data.
    features: list of str
        List of feature columns. The remaining columns are treated as metadata.
    by: list of str
        Columns to groupby during processing, affecting how the features are treated.
    targets: list of str
        Column names to specify as target features.
    meta: dict
        Feature metadata.
    normalizers: dict
        Organize normalization objects with different keys, e.g.,
        having separate normalizers for keys "features" and "targets".
    normalized: dict
        Track for each normalizer whether normalization has been performed.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: Optional[Union[str, List[str]]],  # pylint: disable=invalid-name
        targets: Optional[Union[str, List[str]]] = None,
        force_types: Optional[dict] = None,
    ):
        """Init."""
        # Check data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature data must be a pandas.DataFrame.")

        # Force a range index
        data = to_range_index(data)

        self.by = to_list_optional(by)  # pylint: disable=invalid-name

        if self.by is not None:
            has_columns(data, self.by, raise_error=True)

        feature_list = to_list(features)
        target_list = to_list_optional(targets, none_to_empty=True)

        has_columns(data, feature_list, raise_error=True)
        has_columns(data, target_list, raise_error=True)

        # Add targets to the list of features if they were not included
        self.features = list(set(feature_list + target_list))

        # Type checking and inference
        data = normalize_data(data, self.features)

        self._data = data
        self.meta: Dict[str, FeatureMeta] = {}
        self._infer_feature_types(force_types=force_types)

        self.normalizers: Dict[str, GroupbyNormalizer] = {}
        self.normalized: Dict[str, bool] = {}

    def get_data(
        self, to_indicators: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Get the features data.

        Parameters
        ----------
        to_indicators: str or list of str, optional
            Ordinal features to convert to categorical indicators.

        Returns
        -------
        pandas.DataFrame
            Features data.

        """
        to_indicators = to_list_optional(to_indicators)

        if to_indicators is not None:
            return self._ordinal_to_indicators(to_indicators, inplace=False)

        return self._data

    @property
    def columns(self) -> List[str]:
        """Access as attribute, data columns.

        Returns
        -------
        list of str
            List of all column names.

        """
        return self._data.columns

    def feature_names(
        self,
        feature_type: Optional[str] = None,
        target: Optional[bool] = None,
    ) -> List[str]:
        """Access as attribute, feature names.

        Parameters
        ----------
        feature_type: str, optional
            Filter by feature type.
        target: bool, optional
            Filter by whether the feature is a target.

        Returns
        -------
        list of str
            List of the desired feature names.

        """
        features = self.features

        if feature_type is not None:
            features = [
                col for col in features if self.meta[col].get_type() == feature_type
            ]

        if target is not None:
            features = [col for col in features if self.meta[col].is_target()]

        return features

    @property
    def types(self) -> dict:
        """Access as attribute, feature type names.

        Note: These are framework-specific feature names.

        Returns
        -------
        dict
            Feature type mapped for each feature.

        """
        return {name: meta.get_type() for name, meta in self.meta.items()}

    @property
    def targets(self) -> List[str]:
        """Access as attribute, names of target features.

        Returns
        -------
        list of str
            Names of target features.

        """
        return [col for col, meta in self.meta.items() if meta.is_target()]

    def _update_meta(self, meta_update: dict) -> None:
        """Update feature metadata.

        Parameters
        ----------
        meta_update: dict
            A dictionary in which the values will add/update
            the existing feature metadata dictionary.

        """
        for col, info in meta_update.items():
            if col in self.meta:
                self.meta[col].update(list(info.items()))
            else:
                self.meta[col] = FeatureMeta(**info)

    def _to_feature_types(self, new_types: dict, inplace: bool = True) -> pd.DataFrame:
        """Convert feature types.

        Parameters
        ----------
        new_types: dict
            A map from the feature name to the new feature type.
        inplace: bool
            Whether to perform in-place, or to simply return the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The features data with the relevant conversions.

        """
        invalid = set(new_types.keys()) - set(self.features)
        if len(invalid) > 0:
            raise ValueError(f"Unrecognized features: {', '.join(invalid)}")

        for col, new_type in new_types.items():
            # Check that we aren't converting any categorical indicators to other things
            if col in self.meta:
                if self.meta[col].get_type() == CATEGORICAL_INDICATOR:
                    raise ValueError(
                        "Categorical indicators cannot be converted to other types."
                    )

                if inplace:
                    # Remove original category column if converting to indicators
                    if new_type == CATEGORICAL_INDICATOR:
                        del self.meta[col]
                        self.features.remove(col)

        data, meta = to_types(self._data, new_types)

        if inplace:
            # Append any new indicator features
            for col, fmeta in meta.items():
                if FEATURE_INDICATOR_ATTR in fmeta:
                    self.features.append(col)

            self._data = data
            self._update_meta(meta)

        return data

    def _infer_feature_types(
        self,
        force_types: Optional[dict] = None,
    ):
        """Infer feature types.

        Can optionally force certain types on specified features.

        Parameters
        ----------
        force_types: dict
            A map from the feature name to the forced feature type.

        """
        infer_features = set(self.features) - set(
            to_list_optional(force_types, none_to_empty=True)
        )

        new_types = infer_types(self._data, infer_features)

        # Force certain features to be specific types
        if force_types is not None:
            for feature, type_ in force_types.items():
                new_types[feature] = type_

        self._to_feature_types(new_types)

    def add_normalizer(
        self,
        key: str,
        normalizer: GroupbyNormalizer,
    ) -> None:
        """Add a normalization object.

        Parameters
        ----------
        key: str
            Unique name of the normalizer.
        normalizer: GroupbyNormalizer
            Normalization object.

        """
        if key in self.normalizers:
            raise ValueError(
                "A normalizer with this key already exists. Consider first removing it."
            )

        by = normalizer.get_by()  # pylint: disable=invalid-name
        if by is not None:
            has_columns(self._data, by, raise_error=True)

        normalizer_map = normalizer.get_map()
        features = set(normalizer_map.keys())

        # Check to make sure none of the feature exist in another normalizer
        for norm_key, norm in self.normalizers.items():
            norm_set = set(norm.keys())
            intersect = norm_set.intersection(features)
            if len(intersect) != 0:
                raise ValueError(
                    f"Features {', '.join(intersect)} exist in normalizer {norm_key}."
                )

        # Check for non-existent columns in the map
        nonexistent = set(normalizer_map.keys()) - set(self.features)
        if len(nonexistent) > 0:
            raise ValueError(
                f"The following columns are not features: {', '.join(nonexistent)}."
            )

        # Check for invalid non-numeric columns
        is_numeric = [
            self.meta[col].get_type() == NUMERIC for col in normalizer_map.keys()
        ]
        if not all(is_numeric):
            raise ValueError(
                "Only numeric features may be normalized. Confirm feature choice/type."
            )

        gbn = GroupbyNormalizer(normalizer_map, by)
        gbn.fit(self._data)
        self.normalizers[key] = gbn
        self.normalized[key] = False

    def remove_normalizer(self, key: str) -> None:
        """Remove a normalization object.

        Parameters
        ----------
        key: str
            Unique name of the normalizer.

        """
        if key not in self.normalizers:
            raise ValueError("No normalizer with this key exists.")

        del self.normalizers[key]

    def normalize(self, key: str, inplace: bool = True) -> pd.DataFrame:
        """Normalize.

        Parameters
        ----------
        key: str
            Unique name of the normalizer.
        inplace: bool
            Whether to perform in-place, or to simply return the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The normalized features data.

        """
        if self.normalized[key]:
            raise ValueError(f"Cannot normalize {key}. It has already been normalized.")

        gbn = self.normalizers[key]
        data = gbn.transform(self._data)

        if inplace:
            self._data = data
            self.normalized[key] = True

        return data

    def inverse_normalize(self, key: str, inplace: bool = True) -> pd.DataFrame:
        """Inverse normalize.

        Parameters
        ----------
        key: str
            Unique name of the normalizer.
        inplace: bool
            Whether to perform in-place, or to simply return the DataFrame.

        Returns
        -------
        pandas.DataFrame
            The inversely normalized data.

        """
        if not self.normalized[key]:
            raise ValueError(
                f"Cannot inverse normalize {key}. It has not been normalized."
            )

        gbn = self.normalizers[key]
        data = gbn.inverse_transform(self._data)

        if inplace:
            self._data = data
            self.normalized[key] = False

        return data

    def _ordinal_to_indicators(
        self,
        features: Union[str, List[str]],
        inplace: bool = True,
    ) -> pd.DataFrame:
        """Convert ordinal features to binary categorical indicators.

        Parameters
        ----------
        feautres: str or list of str, optional
            Ordinal features to convert. If not provided, convert all ordinal features.
        inplace: bool
            Whether to perform in-place, or to simply return the DataFrame.

        Returns
        -------
        pandas.DataFrame
            Features data with the relevant conversions.

        """
        for feat in features:
            if feat not in self.features:
                raise ValueError(f"Feature {feat} does not exist.")

            if self.meta[feat].get_type() != ORDINAL:
                raise ValueError(
                    f"{feat} must be an ordinal feature to convert to indicators."
                )

        # Map back to original values
        for feat in features:
            self._data[feat] = (
                self._data[feat].astype(int).replace(self.meta[feat].get_mapping())
            )

        return self._to_feature_types(
            {feat: CATEGORICAL_INDICATOR for feat in features}, inplace=inplace
        )

    def save(self, save_path: str, file_format: str = "parquet") -> None:
        """Save feature data to file.

        Parameters
        ----------
        dataframe: pandas.DataFrame
            Dataframe to save.
        save_path: str
            Path where the file will be saved.
        file_format: str
            File format of the file to save.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        return save_dataframe(self._data, save_path, file_format)


class TabularFeatures(Features):
    """Tabular features."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: Optional[str] = None,
        targets: Optional[Union[str, List[str]]] = None,
        force_types: Optional[dict] = None,
    ):
        """Init."""
        if by is not None and not isinstance(by, str):
            raise ValueError(
                "Tabular features must have no index, or one index input as a string."
            )

        super().__init__(
            data,
            features,
            by,
            targets=targets,
            force_types=force_types,
        )

    def plot_features(
        self,
        features: Optional[Union[str, list]] = None,
    ) -> None:
        """Plot features.

        High-level plotting function for features.

        Parameters
        ----------
        features: str or list of str, optional
            Names of features to plot.

        """
        if features is None:
            plot_histogram(self._data, self.feature_names)
        else:
            plot_histogram(self._data, features)


class TemporalFeatures(Features):
    """Temporal features."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: Union[str, List[str]],
        timestamp_col: str,
        targets: Optional[Union[str, List[str]]] = None,
        force_types: Optional[dict] = None,
        aggregator: Optional[Aggregator] = None,
    ):
        """Init."""
        super().__init__(
            data,
            features,
            by,
            targets=targets,
            force_types=force_types,
        )

        self.timestamp_col = timestamp_col
        self.aggregator = aggregator
        self._check_aggregator()

    def _check_aggregator(self):
        if self.aggregator.get_timestamp_col() != self.timestamp_col:
            raise ValueError(
                "Features and aggregator timestamp columns must be the same."
            )

    def plot_features(
        self,
        features: Optional[Union[str, list]] = None,
    ) -> None:
        """Plot features.

        High-level plotting function for features.

        Parameters
        ----------
        features: list or str, optional
            Names of features to plot.

        """
        if features is None:
            plot_temporal_features(self._data, self.feature_names)
        else:
            plot_temporal_features(self._data, features)

    def aggregate(self) -> pd.DataFrame:
        """Aggregate the data.

        Returns
        -------
        pandas.DataFrame
            Aggregated data.

        """
        if self.aggregator is None:
            raise ValueError(
                "Must pass an aggregator when creating features to aggregate."
            )

        agg_features = self.aggregator.get_aggfuncs().keys()

        # Check for non-existent columns in the map
        nonexistent = set(agg_features) - set(self.features)
        if len(nonexistent) > 0:
            raise ValueError(
                f"The following columns are not features: {', '.join(nonexistent)}."
            )

        return self.aggregator(self._data)
