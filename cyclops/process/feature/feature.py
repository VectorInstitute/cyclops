"""Feature processing."""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cyclops.plotter import plot_histogram, plot_temporal_features
from cyclops.process.aggregate import Aggregator
from cyclops.process.constants import (
    BINARY,
    CATEGORICAL_INDICATOR,
    FEATURE_INDICATOR_ATTR,
    FEATURE_MAPPING_ATTR,
    FEATURE_META_ATTR_DEFAULTS,
    FEATURE_META_ATTRS,
    FEATURE_TARGET_ATTR,
    FEATURE_TYPE_ATTR,
    FEATURE_TYPES,
    FEATURES,
    NUMERIC,
    ORDINAL,
)
from cyclops.process.feature.handle_types import infer_types, normalize_data, to_types
from cyclops.process.feature.normalize import GroupbyNormalizer
from cyclops.process.feature.split import split_idx
from cyclops.process.feature.vectorized import Vectorized
from cyclops.process.util import has_columns, has_range_index, to_range_index
from cyclops.utils.common import to_list, to_list_optional
from cyclops.utils.file import save_dataframe
from cyclops.utils.log import setup_logging

# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


# mypy: ignore-errors


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
        targets: Union[str, List[str], None] = None,
        force_types: Optional[dict] = None,
        normalizers: Optional[Dict[str, GroupbyNormalizer]] = None,
    ):
        """Init."""
        # Check data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature data must be a pandas.DataFrame.")

        if not has_range_index(data):
            raise ValueError(
                "Data required to have a range index. Try resetting the index."
            )

        # Force a range index
        data = to_range_index(data)

        feature_list = to_list(features)
        target_list = to_list_optional(targets, none_to_empty=True)

        if len(feature_list) == 0:
            raise ValueError("Must specify at least one feature.")

        has_columns(data, feature_list, raise_error=True)
        has_columns(data, target_list, raise_error=True)

        self.by = to_list_optional(by)  # pylint: disable=invalid-name

        if self.by is not None:
            has_columns(data, self.by, raise_error=True)

            if len(set(self.by).intersection(set(feature_list))) != 0:
                raise ValueError("Columns in 'by' cannot be considered features.")

        # Add targets to the list of features if they were not included
        self.features = list(set(feature_list + target_list))

        # Type checking and inference
        data = normalize_data(data, self.features)

        self.data = data
        self.meta: Dict[str, FeatureMeta] = {}
        self._infer_feature_types(force_types=force_types)

        self.normalizers: Dict[str, GroupbyNormalizer] = {}
        self.normalized: Dict[str, bool] = {}
        if normalizers is not None:
            for key, normalizer in normalizers.items():
                self.add_normalizer(key, normalizer)

    def get_data(
        self,
        features_only: bool = True,
        to_binary_indicators: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """Get the features data.

        Parameters
        ----------
        features_only: bool, default = True
            Whether to return only the by and feature columns.
        to_indicators: str or list of str, optional
            Ordinal features to convert to categorical indicators.

        Returns
        -------
        pandas.DataFrame
            Features data.

        """
        data = self.data

        # Take only the feature columns
        if features_only:
            data = data[self.features + to_list_optional(self.by, none_to_empty=True)]

        # Convert to binary categorical indicators
        if to_binary_indicators is not None:
            data = self._ordinal_to_indicators(data, to_list(to_binary_indicators))

        # Convert binary columns from boolean to integer
        binary_cols = [col for col, value in self.types.items() if value == BINARY]
        for col in binary_cols:
            data[col] = data[col].astype(int)

        return data.set_index(self.by)

    @property
    def columns(self) -> List[str]:
        """Access as attribute, data columns.

        Returns
        -------
        list of str
            List of all column names.

        """
        return self.data.columns

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
                col
                for col in self.features
                if self.meta[col].get_type() == feature_type
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

    def features_by_type(self, type_: str) -> List[str]:
        """Get feature names of a given type.

        Parameters
        ----------
        type_: str
            Feature type.

        Returns
        -------
        list of str
            Names of all features with the given type.

        """
        return [name for name, ftype in self.types.items() if ftype == type_]

    def split_by_values(self, value_splits: List[np.ndarray]) -> Tuple:
        """Split the data into multiple datasets by values.

        Parameters
        ----------
        value_splits: list of numpy.ndarray
            A list with an element for each split, where the elements are numpy.ndarray
            with values determined how the splits are segmented.

        Returns
        -------
        tuple
            A tuple of Features objects with the split data.

        """
        on_col = self.by[0]
        unique = self.data[on_col].unique()
        unique.sort()

        all_vals = np.concatenate(value_splits)
        all_vals.sort()

        if not np.array_equal(unique, all_vals):
            raise ValueError("Invalid split values.")

        datas = []
        for split in value_splits:
            data_copy = self.data.copy()
            datas.append(data_copy[data_copy[on_col].isin(split)])

        save_data = self.data
        self.data = None
        splits = [copy.deepcopy(self) for _ in range(len(value_splits))]
        for i, split in enumerate(splits):
            split.data = datas[i]
        self.data = save_data

        return tuple(splits)

    def split(
        self,
        fractions: Union[float, List[float]] = 1.0,
        randomize: bool = True,
        seed: int = None,
    ):
        """Split the data into multiple datasets by fractions.

        Parameters
        ----------
        fractions: list of float or float, optional
            Fraction(s) of samples between 0 and 1 to use for each split.
        randomize: bool, default = True
            Whether to randomize the data in the splits.
        seed: int, optional
            Seed for random number generator.

        Returns
        -------
        tuple
            A tuple of Features objects with the split data.

        """
        value_splits = self.compute_value_splits(fractions, randomize, seed)
        return self.split_by_values(value_splits)

    def compute_value_splits(
        self,
        fractions: Union[float, List[float]] = 1.0,
        randomize: bool = True,
        seed: int = None,
    ) -> List[np.ndarray]:
        """Compute the value splits given fractions.

        Parameters
        ----------
        fractions: list of float or float, optional
            Fraction(s) of samples between 0 and 1 to use for each split.
        randomize: bool, default = True
            Whether to randomize the data in the splits.
        seed: int, optional
            Seed for random number generator.

        Returns
        -------
        list of numpy.ndarray
            A list with an element for each split, where the elements are numpy.ndarray
            with values determined how the splits are segmented.

        """
        on_col = self.by[0]
        unique = self.data[on_col].unique()
        unique.sort()
        idx_splits = list(
            split_idx(fractions, len(unique), randomize=randomize, seed=seed)
        )
        value_splits = [np.take(unique, split) for split in idx_splits]
        return value_splits

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

    def _to_feature_types(
        self,
        data: pd.DataFrame,
        new_types: dict,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """Convert feature types.

        Parameters
        ----------
        data: pandas.DataFrame
            Features data.
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
            if col in self.meta:
                # Do not allow converting to categorical indicators inplace
                if inplace and new_type == CATEGORICAL_INDICATOR:
                    raise ValueError(
                        f"Cannot convert {col} to binary categorical indicators."
                    )

        data, meta = to_types(data, new_types)

        if inplace:
            # Append any new indicator features
            for col, fmeta in meta.items():
                if FEATURE_INDICATOR_ATTR in fmeta:
                    self.features.append(col)

            self.data = data
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
        infer_features = to_list(
            set(self.features) - set(to_list_optional(force_types, none_to_empty=True))
        )

        new_types = infer_types(self.data, infer_features)

        # Force certain features to be specific types
        if force_types is not None:
            for feature, type_ in force_types.items():
                new_types[feature] = type_

        self._to_feature_types(self.data, new_types)

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
            has_columns(self.data, by, raise_error=True)

        normalizer_map = normalizer.get_map()
        features = set(normalizer_map.keys())

        # Check to make sure none of the feature exist in another normalizer
        for norm_key, norm in self.normalizers.items():
            norm_set = set(norm.normalizer_map.keys())
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
        gbn.fit(self.data)
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
        data = gbn.transform(self.data)

        if inplace:
            self.data = data
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
        data = gbn.inverse_transform(self.data)

        if inplace:
            self.data = data
            self.normalized[key] = False

        return data

    def _ordinal_to_indicators(
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
    ) -> pd.DataFrame:
        """Convert ordinal features to binary categorical indicators.

        Parameters
        ----------
        data: pandas.DataFrame
            Features data.
        features: str or list of str, optional
            Ordinal features to convert. If not provided, convert all ordinal features.

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
            mapping = self.meta[feat].get_mapping()
            data[feat].replace(mapping, inplace=True)

        # Convert to binary categorical indicators
        return self._to_feature_types(
            data, {feat: CATEGORICAL_INDICATOR for feat in features}, inplace=False
        )

    def save(self, save_path: str, file_format: str = "parquet") -> str:
        """Save data to file.

        Parameters
        ----------
        save_path: str
            Path where the file will be saved.
        file_format: str
            File format of the file to save.

        Returns
        -------
        str
            Processed save path for upstream use.

        """
        return save_dataframe(self.data, save_path, file_format=file_format)

    def slice(
        self,
        slice_map: Dict[str, Union[Any, List[Any]]] = None,
        slice_query: Optional[str] = None,
        replace: bool = False,
    ) -> np.ndarray:
        """Slice the data across column(s), given values.

        Parameters
        ----------
        slice_map: Dict, optional
            Dictionary with column name(s) as keys, and value or list of values
            to filter on as values.
        slice_query: str, optional
            A string to specify conditions that uses the Pandas DataFrame query API.
            If specified along with slice_map, the slice_map is applied first.
        replace: bool, optional
            If set to True, the data is replaced with the sliced data, and the
            the values (by column) of the sliced dataset are returned.

        Returns
        -------
        np.ndarray
            Array of the values of the by column, in the sliced dataset.

        """
        sliced_indices = []
        if not slice_map:
            slice_map = {}
        for slice_col, slice_vals in slice_map.items():
            sliced_indices.append(
                self.data[self.data[slice_col].isin(to_list(slice_vals))][
                    self.by[0]
                ].values
            )
        if sliced_indices:
            intersect_indices = set.intersection(*map(set, sliced_indices))
            sliced_data = self.data[self.data[self.by[0]].isin(intersect_indices)]
        else:
            sliced_data = self.data

        if slice_query:
            sliced_data = sliced_data.query(slice_query)

        if replace:
            self.data = sliced_data

        return sliced_data[self.by[0]].values


class TabularFeatures(Features):
    """Tabular features."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: str,
        targets: Optional[Union[str, List[str]]] = None,
        force_types: Optional[dict] = None,
    ):
        """Init."""
        if not isinstance(by, str):
            raise ValueError(
                "Tabular features index input as a string representing a column."
            )

        super().__init__(
            data,
            features,
            by,
            targets=targets,
            force_types=force_types,
        )

    def vectorize(self, **get_data_kwargs) -> Vectorized:
        """Vectorize the tabular data.

        Parameters
        ----------
        **get_data_kwargs
            Keyword arguments to be fed to get_data.

        Returns
        -------
        tuple
            (data, by_map, feat_map), (pandas.DataFrame, dict, dict)
            feat_map is the feature order and by_map is the order of
            the by column, or None if no by was provided.

        """
        if "features_only" in get_data_kwargs:
            raise ValueError(
                "Cannot specify 'features_only'. It will be set to True by default."
            )

        get_data_kwargs["features_only"] = True

        data = self.get_data(**get_data_kwargs).reset_index()

        by_map = list(data[self.by[0]].values)
        data = data.drop(self.by, axis=1)
        feat_map = list(data.columns)
        return Vectorized(
            data.values, indexes=[by_map, feat_map], axis_names=[self.by[0], FEATURES]
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
            plot_histogram(self.data, self.feature_names())
        else:
            plot_histogram(self.data, features)


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
            plot_temporal_features(self.data, self.feature_names())
        else:
            plot_temporal_features(self.data, features)

    def aggregate(self, **aggregate_kwargs) -> pd.DataFrame:
        """Aggregate the data.

        Parameters
        ----------
        **aggregate_kwargs
            Keywords to pass to the aggregation function.

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

        return self.aggregator(self.data, **aggregate_kwargs)


def split_features(
    features: List[Union[Features, TabularFeatures, TemporalFeatures]],
    fractions: Optional[Union[float, List[float]]] = None,
    randomize: bool = True,
    seed: int = None,
) -> Tuple:
    """Split a set of features using the same uniquely identifying values.

    Parameters
    ----------
    features: list of Features or TabularFeatures or TemporalFeatures
        List of feature objects.
    fractions: list, optional
        Fraction(s) of samples between 0 and 1 to use for each split.
    randomize: bool, default = True
        Whether to randomize the data in the splits.
    seed: int, optional
        Seed for random number generator.

    Returns
    -------
    tuple
        A tuple of the dataset splits, where each contains a tuple of splits.
        e.g., split1, split2 = split_features([features1, features2], 0.5)
        train1, test1 = split1
        train2, test2 = split2

    """
    value_splits = features[0].compute_value_splits(
        fractions, randomize=randomize, seed=seed
    )
    feature_splits = [feat.split_by_values(value_splits) for feat in features]
    return tuple(feature_splits)
