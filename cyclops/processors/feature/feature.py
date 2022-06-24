"""Feature handling for automatic feature creation from processed data."""
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
from cyclops.processors.util import has_columns
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
    target: bool, default = False
        Is the feature a target variable, True if yes, else False.
    indicator_of: str, optional
        If not None, the column from which this indicator was generated.
    allowed_values: numpy.ndarray, optional
        The feature values allowed. If None, allow any values.

    """

    def __init__(self, **kwargs) -> None:
        """Initialize."""
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
        """Get the name of an indicator's original categorical column.

        Returns
        -------
        str or None
            The categorical column from which an indicator was generated, or None if
            not a categorical indicator.

        """
        return getattr(self, FEATURE_INDICATOR_ATTR)

    def get_mapping(self) -> Optional[dict]:
        """Get the mapping binary and ordinal categories.

        Returns
        -------
        int or None
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
        having separate normalizers for "features" and "targets".
    normalized: dict
        Track for each normalizer whether normalization has been performed.

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: Optional[Union[str, List[str]]],  # pylint: disable=invalid-name
        targets: Optional[Union[str, List[str]]] = None,
        to_indicators: bool = False,
        force_types: Optional[dict] = None,
    ):
        """Init."""
        # Check data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature data must be a pandas.DataFrame.")

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
        
        self.data = data
        self.meta: Dict[str, FeatureMeta] = {}
        self._infer_feature_types(
            to_indicators=to_indicators, force_types=force_types
        )

        self.normalizers: Dict[str, GroupbyNormalizer] = {}
        self.normalized: Dict[str, bool] = {}

    @property
    def columns(self) -> List[str]:
        """Access as attribute, data columns.

        Returns
        -------
        list
            List of all feature names.

        """
        return self.data.columns

    @property
    def feature_names(self) -> List[str]:
        """Access as attribute, feature names.

        Returns
        -------
        list
            List of all feature names.

        """
        return self.features

    @property
    def types(self) -> dict:
        """Access as attribute, feature type names.

        Note: These are built-in feature names.

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

    def _to_feature_types(self, new_types: dict) -> None:
        """Manually convert feature types.

        Parameters
        ----------
        types: dict
            A map from the column name to the feature type.

        """
        # Check that we aren't converting any categorical indicators to other things
        for col, new_type in new_types.items():
            if col in self.meta:
                if self.meta[col].get_type() == CATEGORICAL_INDICATOR:
                    raise ValueError(
                        "Categorical indicators cannot be converted to other types."
                    )

                # Remove original category column if converting to indicators
                if new_type == CATEGORICAL_INDICATOR:
                    del self.meta[col]

        # Edit existing data
        for col, new_type in new_types.items():
            # Remove any existing metadata
            if col in self.meta:
                del self.meta[col]
            
            # Remove original category column if converting to indicators
            if new_type == CATEGORICAL_INDICATOR:
                self.features.remove(col)

        data, meta = to_types(self.data, new_types)

        # Append any new indicator features
        for col, fmeta in meta.items():
            if FEATURE_INDICATOR_ATTR in fmeta:
                self.features.append(col)

        self.data = data
        self._update_meta(meta)

    def _infer_feature_types(
        self,
        to_indicators: bool = True,
        force_types: Optional[dict] = None,
    ):
        """Infer feature types."""
        new_types = infer_types(
            self.data, self.features, to_indicators=to_indicators
        )
        
        # Force certain features to be specific types
        if force_types is not None:
            invalid = set(force_types.keys()) - set(self.features)
            if len(invalid) > 0:
                raise ValueError(f"Unrecognized features: {', '.join(invalid)}")
            
            for feature, type_ in force_types.items():
                new_types[feature] = type_
        
        self._to_feature_types(new_types)

    def add_normalizer(
        self,
        key: str,
        normalizer: GroupbyNormalizer,
    ) -> None:
        """Create and store a normalization object.

        Parameters
        ----------
        key: str
            Unique name of the normalizer.
        normalizer_map: dict
            A map from the column name to the type of normalization, e.g.,
            "event_values": "standard"
        by: str or list of str, optional
            Columns to use in the groupby, affecting how values are normalized.

        """
        if key in self.normalizers:
            raise ValueError(
                "A normalizer with this key already exists. Consider first removing it."
            )

        by = normalizer.get_by()
        if by is not None:
            has_columns(self.data, by, raise_error=True)
        
        
        normalizer_map = normalizer.get_map()
        features = set(normalizer_map.keys())
        
        # Check to make sure none of the feature exist in another normalizer
        for key, norm in self.normalizers.items():
            norm_set = set(norm.keys())
            intersect = norm_set.intersection(features)
            if len(intersect) != 0:
                raise ValueError(
                    f"Features {', '.join(intersect)} exist in another normalizer."
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
                "Only numeric features may be normalized. Confirm feature choice and type."
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
            Whether to perform in-place, or to return the DataFrame.

        Returns
        -------
        pandas.DataFrame or None
            If inplace is True, returns the normalized data.

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
            Whether to perform in-place, or to return the DataFrame.

        Returns
        -------
        pandas.DataFrame or None
            If inplace is True, returns the inverse normalized data.

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

    def ordinal_to_categorical(
        self,
        features: Optional[Union[str, List[str]]] = None
    ) -> None:
        """Convert ordinal features to binary categorical indicators.
        
        Parameters
        ----------
        feautres: str or list of str, optional
            Ordinal features to convert. If not provided, convert all ordinal features.
        
        """
        features = to_list_optional(features)
        
        if features is None:
            features = [f for f in self.features if self.meta[f].get_type() == ORDINAL]
        
        for feat in features:
            if not feat in self.features:
                raise ValueError(f"Feature {feat} does not exist.")
            
            if self.meta[feat].get_type() != ORDINAL:
                raise ValueError(f"{feat} is not an ordinal feature.")
        
        # Map back to original values
        for feat in features:
            self.data[feat] = self.data[feat].astype(int).replace(self.meta[feat].get_mapping())
        
        self._to_feature_types({feat: CATEGORICAL_INDICATOR for feat in features})

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
        return save_dataframe(self.data, save_path, file_format)


class TabularFeatures(Features):
    """Tabular features."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: Optional[str] = None,
        targets: Optional[Union[str, List[str]]] = None,
        to_indicators: bool = False,
        force_types: Optional[dict] = None,
    ):
        """Initialize."""
        if by is not None and not isinstance(by, str):
            raise ValueError(
                "Tabular features must have no index, or one index input as a string."
            )

        super().__init__(
            data,
            features,
            by,
            targets=targets,
            to_indicators=to_indicators,
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
            plot_histogram(self.data, self.feature_names)
        else:
            plot_histogram(self.data, features)


class TemporalFeatures(Features):
    """Temporal features."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        features: Union[str, List[str]],
        by: Union[str, List[str]],
        targets: Optional[Union[str, List[str]]] = None,
        to_indicators: bool = False,
        force_types: Optional[dict] = None,
        aggregator: Optional[Aggregator] = None,
    ):
        """Initialize."""
        super().__init__(
            data,
            features,
            by,
            targets=targets,
            to_indicators=to_indicators,
            force_types=force_types,
        )

        self.aggregator = aggregator

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
            plot_temporal_features(self.data, self.feature_names)
        else:
            plot_temporal_features(self.data, features)
