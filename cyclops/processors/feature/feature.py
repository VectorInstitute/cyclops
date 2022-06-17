"""Feature handling for automatic feature creation from processed data."""

"""
Still need to do imputation...

DON'T FORGET TO REMOVE ORIGINAL COLUMN AFTER CREATING CATEGORICAL INDICATORS

No longer storing all features as a single object, as this didn't make sense.
Temporal/tabular features go through different parts of the pipeline anyway,
and this should be done asynchronously in separate object instances

No more loading directly into the object. You have to load the data and then create the object again.
This makes more sense, because there's too much other information that needs to be specified, not just the data.

There also might be different temporal features you might want to process separately, e.g., two TemporalFeatures

Columns we want to group by during processing is probably not the same as what we want to groupby during something like normalization
E.g., for temporal
Processing: ENCOUNTER_ID, EVENT_NAME, TIMESTEP
Normalization: EVENT_NAME

It is the exact same deal for the type checking. I feel like we actually want to do it over all encounters,
which means we need to remove the encounters from the groupby

We won't have performed aggregation by this point, so it should just be ENCOUNTERS, EVENT NAMES in the by argument, whereas we just want EVENT NAMES


# INCORPORATE encounter_id FILTERING into plot_features in TemporalFeatures


# NO MORE "ADDING" OR REMOVING OF FEATURES
# If need be, we can create a merge in Features with another Features object,
# but this limits all of the feature handling to be in the init

# MAKE SURE THAT A TARGET CATEGORICAL TURNED DUMMY INDICATORS
# MAKE SURE THE INDICATORS ARE TARGETS

"""
import logging
from typing import Any, Dict, List, Optional, Union
import os

import collections.abc

import numpy as np
import pandas as pd

from codebase_ops import get_log_file_path
from cyclops.constants import FEATURES
from cyclops.plotter import plot_histogram, plot_temporal_features

from cyclops.utils.file import save_dataframe

from cyclops.processors.aggregate import Aggregator
from cyclops.processors.constants import (
    GROUP,
    MIN_MAX,
    MISSING_CATEGORY,
    STANDARD,
    NUMERIC,
    FEATURE_TYPES
)
from cyclops.processors.feature.normalization import GroupbyNormalizer
from cyclops.processors.impute import Imputer, impute_features
from cyclops.processors.util import is_timeseries_data, has_columns
from cyclops.utils.common import to_list
from cyclops.utils.log import setup_logging
from cyclops.processors.feature.type_handling import (
    normalize_types,
    detect_types,
    infer_types,
    to_types,
)

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
        If not None, indicates the original column from which this indicator was created.
    allowed_values: numpy.ndarray, optional
        The feature values allowed. If None, allow any values.
    """
    def __init__(self,
        type_: str,
        target: bool = False,
        indicator_of: Optional[str] = None,
    ):
        """Initialize."""
        if type_ not in FEATURE_TYPES:
            raise ValueError(
                f"Feature type '{type_}' not in {', '.join(FEATURE_TYPES)}."
            )
        
        self.type_ = type_
        self.target = target
        self.indicator_of = indicator_of
            
    
    def get_type(self) -> str:
        """Return the feature type.
        
        Returns
        -------
        str
            Feature type.
        """
        return self.type_

    def is_target(self) -> bool:
        """Return whether the feature is a target.
        
        Returns
        -------
        bool
            Whether the feature is a target.
        """
        return self.target

    def _update(self, meta: List[tuple]) -> None:
        """Updates meta attributes.
        
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
    by: str or list of str
        Columns to groupby during processing, affecting how the features are treated.
    targets: str or list of str
        Column names to specify as target features.
    meta: dict
        Feature metadata.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        by: Union[str, List[str]],
        targets: Union[str, List[str]] = [],
        infer_feature_types: bool = True,
        allow_categorical: bool = True,
    ):
        """Initalize."""
        # Check data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Feature data must be a pandas.DataFrame.")
        
        has_columns(data, by, raise_error=True)
        has_columns(data, targets, raise_error=True)

        
        # Type checking and conversions
        data = normalize_types(data)
        types = detect_types(data)
        
        if infer_feature_types:
            data, types = infer_types(data, types, allow_categorical=allow_categorical)
        
        self.data = data.set_index(by)
        self.by = to_list(by)
        
        # Create metadata
        self.meta = {}
        for col in data:
            self.meta[col] = FeatureMeta(
                types[col],
                target=col in targets,
                indicator_of=None
            )
        
        self.normalizers = {}

    @property
    def names(self) -> List[str]:
        """Access as attribute, feature names.

        Returns
        -------
        list
            List of all feature names.

        """
        return self.data.columns

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
        return [col for col in self.meta if self.meta[col].is_target()],

    def _update_meta(self, meta_update: dict) -> None:
        """Update feature metadata.
        
        Parameters
        ----------
        meta_update: dict
            A dictionary in which the values will add/update
            the existing feature metadata dictionary.

        """        
        for col, info in meta_update.items():
            self.meta[col]._update(list(info.items()))

    def to_feature_types(self, types: dict) -> None:
        """Manually convert feature types.
        
        Parameters
        ----------
        types: dict
            A map from the column name to the feature type.

        """
        self.data, updated_meta = to_types(self.data, types)
        self._update_meta(updated_meta)
    
    def infer_feature_types(self):
        """Infer feature types.
        
        """
        self.data, updated_meta = infer_types(self.data, self.types)
        self._update_meta(updated_meta)
    
    def create_normalizer(
        self,
        key: str,
        normalizer_map: dict,
        by: Optional[Union[str, List[str]]] = None
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
            raise ValueError("A normalizer with this key already exists. Consider first removing it.")
        
        is_numeric = [self.meta[col].get_type() == NUMERIC for col in normalizer_map.keys()]
        if not all(is_numeric):
            raise ValueError("Only numeric features may be normalized. Check the feature types.")
        
        gbn = GroupbyNormalizer(normalizer_map, by)
        gbn.fit(self.data)
        self.normalizers[key] = gbn
    
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

    def normalize(
        self,
        key: str,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
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
        gbn = self.normalizers[key]
        data = gbn.transform(self.data)
        
        if inplace:
            self.data = data
        else:
            return data
    
    def inverse_normalize(
        self,
        key: str,
        inplace: bool = True
    ) -> Optional[pd.DataFrame]:
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
        data = gbn.transform(self.data)
        
        if inplace:
            self.data = data
        else:
            return data

    def save(
        self,
        save_path: str,
        file_format: str = "parquet"
    ) -> None:
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
    def __init__(
        self,
        data: pd.DataFrame,
        by: Optional[str] = None,
        targets: Union[str, List[str]] = [],
        infer_feature_types: bool = True,
    ):
        """Initialize."""
        if by is not None and not isinstance(by, str):
            raise ValueError(
                "Tabular features must have just one index (must input as string)."
            )
        
        super().__init__(data, by, targets=targets, infer_feature_types=infer_feature_types)

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
            plot_histogram(self.data, self.names)
        else:
            plot_histogram(self.data, features)
        

class TemporalFeatures(Features):
    """Temporal features."""
    def __init__(
        self,
        data: pd.DataFrame,
        by: Union[str, List[str]],
        aggregator: Optional[Aggregator] = None,
        targets: Union[str, List[str]] = [],
        infer_feature_types: bool = True,
    ):
        """Initialize."""
        super().__init__(data, by, targets=targets, infer_feature_types=infer_feature_types)

        self.aggregator = aggregator
    
    def plot_features(
        self,
        encounter_id: int = None,
        features: Optional[Union[str, list]] = None,
    ) -> None:
        """Plot features.

        High-level plotting function for features.

        Parameters
        ----------
        encounter_id: int, optional
            Encounter ID.
        names: list or str, optional
            Names of features to plot.

        """
        if features is None:
            plot_temporal_features(self.data, self.names)
        else:
            plot_temporal_features(self.data, features)
